"""CPU checks for evidence interaction, masking, residuals and frozen weights."""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
from partner_evidence_teacher_model import EvidenceInteractionTeacher


class EvidenceTeacherModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.weight = torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.7, 0.1], [0.3, 0.0, 1.2]])
        self.queries = torch.tensor([[1.0, 0.5, -0.2], [0.0, 1.0, 0.8]])
        self.evidence = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [9., 9., 9.]],
                                      [[0., 0., 1.], [1., 0., 1.], [-1., 0., 0.]],
                                      [[0., 1., 1.], [5., 5., 5.], [6., 6., 6.]]])
        self.mask = torch.tensor([[True, True, False], [True, True, True], [True, False, False]])

    def model(self, mode="attention"):
        return EvidenceInteractionTeacher(self.weight, hidden_dim=4, mlp_hidden=5, mode=mode)

    def activate_residual(self, model):
        with torch.no_grad():
            model.mlp[-1].weight.copy_(torch.linspace(-0.7, 0.6, model.mlp[-1].weight.numel()).reshape_as(
                model.mlp[-1].weight))
            model.mlp[-1].bias.fill_(0.2)

    def test_epoch_zero_matches_frozen_mean_base_and_supplied_means(self):
        clean = self.evidence.masked_fill(~self.mask.unsqueeze(-1), 0.)
        means = clean.sum(1) / self.mask.sum(1, keepdim=True)
        expected = F.normalize(F.linear(self.queries, self.weight), dim=-1) @ F.normalize(
            F.linear(means, self.weight), dim=-1).T
        for mode in ("attention", "mean"):
            model = self.model(mode)
            torch.testing.assert_close(model(self.queries, self.evidence, self.mask), expected, rtol=0, atol=0)
            supplied = means + torch.tensor([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5]])
            candidates = model.prepare_candidates(self.evidence, self.mask, raw_means=supplied)
            actual = model.score_prepared(model.prepare_queries(self.queries), candidates)
            expected_supplied = F.normalize(F.linear(self.queries, self.weight), dim=-1) @ F.normalize(
                F.linear(supplied, self.weight), dim=-1).T
            torch.testing.assert_close(actual, expected_supplied, rtol=0, atol=0)

    def test_padding_cannot_change_scores_or_gradients(self):
        model = self.model()
        self.activate_residual(model)
        changed = self.evidence.clone()
        changed[~self.mask] = float("nan")
        expected = model(self.queries, self.evidence, self.mask)
        actual = model(self.queries, changed, self.mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        differentiable = changed.clone().requires_grad_(True)
        model(self.queries, differentiable, self.mask).sum().backward()
        self.assertTrue(torch.isfinite(differentiable.grad).all())
        self.assertTrue((differentiable.grad[~self.mask] == 0).all())
        changed[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite real evidence"):
            model(self.queries, changed, self.mask)

    def test_evidence_permutations_and_candidate_chunks_preserve_scores(self):
        model = self.model()
        self.activate_residual(model)
        queries = model.prepare_queries(self.queries)
        candidates = model.prepare_candidates(self.evidence, self.mask)
        scores = model.score_prepared(queries, candidates)
        pieces = [model.score_prepared(queries, tuple(part[start:start + 1] for part in candidates))
                  for start in range(len(self.evidence))]
        torch.testing.assert_close(torch.cat(pieces, dim=1), scores)
        order = torch.tensor([2, 0, 1])
        reordered = model(self.queries, self.evidence[order], self.mask[order])
        torch.testing.assert_close(reordered, scores[:, order])
        slots = torch.tensor([2, 0, 1])
        torch.testing.assert_close(model(self.queries, self.evidence[:, slots], self.mask[:, slots]), scores)

    def test_attention_is_query_conditioned_and_diagnostics_are_normalized(self):
        model = EvidenceInteractionTeacher(torch.eye(2), hidden_dim=2, mlp_hidden=3,
                                           attention_temperature=0.5)
        with torch.no_grad():
            model.head.weight.copy_(torch.eye(2))
        evidence = torch.tensor([[[1., 0.], [-1., 0.]], [[0., 1.], [100., 100.]]])
        mask = torch.tensor([[True, True], [True, False]])
        candidates = model.prepare_candidates(evidence, mask)
        queries = model.prepare_queries(torch.tensor([[1., 0.], [0., 1.]]))
        _, stats = model.score_prepared(queries, candidates, return_diagnostics=True)
        probability = torch.tensor([2., -2.]).softmax(0)
        expected_entropy = -(probability * probability.log()).sum() / torch.tensor(2.).log() + 1.
        torch.testing.assert_close(stats["normalized_attention_entropy_sum"], expected_entropy)
        torch.testing.assert_close(stats["attention_max_weight_sum"], probability.max() + 0.5 + 2.)
        self.assertEqual(stats["attention_pair_count"], 4)
        model.mode = "mean"
        _, uniform = model.score_prepared(queries, candidates, return_diagnostics=True)
        torch.testing.assert_close(uniform["normalized_attention_entropy_sum"], torch.tensor(2.))
        torch.testing.assert_close(uniform["attention_max_weight_sum"], torch.tensor(3.))

    def test_modes_have_identical_parameters_but_learn_different_contexts(self):
        attention = self.model("attention")
        mean = self.model("mean")
        mean.load_state_dict(attention.state_dict())
        self.assertEqual(sum(p.numel() for p in attention.parameters()), sum(p.numel() for p in mean.parameters()))
        self.assertEqual(tuple(attention.state_dict()), tuple(mean.state_dict()))
        self.activate_residual(attention)
        mean.load_state_dict(attention.state_dict())
        self.assertFalse(torch.allclose(attention(self.queries, self.evidence, self.mask),
                                        mean(self.queries, self.evidence, self.mask)))

    def test_base_stays_frozen_head_updates_and_residual_is_bounded(self):
        model = self.model()
        original_base = model.base_weight.clone()
        original_head = model.head.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.)
        for _ in range(5):
            optimizer.zero_grad()
            scores = model(self.queries, self.evidence, self.mask)
            loss = -F.log_softmax(scores / 0.1, dim=1)[[0, 1], [1, 2]].mean()
            loss.backward()
            optimizer.step()
        self.assertNotIn("base_weight", dict(model.named_parameters()))
        self.assertIsNone(model.base_weight.grad)
        torch.testing.assert_close(model.base_weight, original_base, rtol=0, atol=0)
        self.assertFalse(torch.equal(model.head.weight, original_head))
        queries, candidates = model.prepare_queries(self.queries), model.prepare_candidates(self.evidence, self.mask)
        base = queries[0] @ candidates[0].T
        with torch.no_grad():
            model.mlp[-1].bias.fill_(100.)
        difference = model.score_prepared(queries, candidates) - base
        self.assertTrue((difference.abs() <= model.residual_scale + 1.e-6).all())
        self.assertTrue((difference > 0.249).all())

    def test_invalid_inputs_rejected_at_preparation(self):
        with self.assertRaisesRegex(ValueError, "square matrix"):
            EvidenceInteractionTeacher(torch.ones(2, 3))
        with self.assertRaisesRegex(ValueError, "temperature"):
            EvidenceInteractionTeacher(torch.eye(2), attention_temperature=0.)
        model = self.model()
        with self.assertRaisesRegex(ValueError, "real evidence"):
            model.prepare_candidates(self.evidence, torch.zeros_like(self.mask))
        with self.assertRaisesRegex(ValueError, "Evidence mask"):
            model.prepare_candidates(self.evidence, self.mask.long())
        with self.assertRaisesRegex(ValueError, "Nonfinite query"):
            model.prepare_queries(torch.full_like(self.queries, float("nan")))
        with self.assertRaisesRegex(ValueError, "candidate means"):
            model.prepare_candidates(self.evidence, self.mask, raw_means=torch.ones(2, 3))


if __name__ == "__main__":
    unittest.main()
