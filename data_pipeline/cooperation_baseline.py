"""CPU-only first-partner recommendation baseline (Python 3.8+, standard library).

Index only pre-cutoff company patent texts in documents.<split>.jsonl.  Gold
cooperation partners are used solely for evaluation, never to compute scores.
Run from src/, for example::

    python ../data_pipeline/cooperation_baseline.py evaluate \
        --data-dir ../data/patent/cooperation_v1 --split test --top-k 20
    python ../data_pipeline/cooperation_baseline.py recommend \
        --data-dir ../data/patent/cooperation_v1 --split test --company CODE

Recall is the mean, over queries, of retrieved positives / all query positives.
MRR uses the first positive's rank in the entire allowed candidate pool.
Unobserved cooperation candidates are unlabeled, not verified negatives.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import sys
import time
import unicodedata


RECALL_CUTOFFS = (5, 10, 20, 50, 100)
TOKENIZER_VERSION = "nfkc_cjk_bigrams_ascii_words_v1"
TEXT_PARTS = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+|[a-z0-9]+")


def tokenize(text):
    """Chinese adjacent bigrams within runs; lowercase ASCII words/numbers.

    Punctuation separates runs, so unrelated words either side of a comma do
    not produce a false bigram. One-character Chinese runs stay searchable.
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    result = []
    for match in TEXT_PARTS.finditer(normalized):
        part = match.group()
        if "\u3400" <= part[0] <= "\ufaff":
            if len(part) == 1:
                result.append(part)
            else:
                result.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            result.append(part)
    return result


def read_jsonl(path):
    """Yield rows without materializing the query file."""
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
            except (ValueError, TypeError) as exc:
                raise ValueError("{}:{}: {}".format(path, line_number, exc)) from exc
            yield row


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query_text(query):
    center = query.get("q_text", "")
    neighbors = query.get("q_n_text", [])
    if not isinstance(center, str) or not isinstance(neighbors, list):
        raise ValueError("q_text must be a string and q_n_text must be a list")
    if any(not isinstance(text, str) for text in neighbors):
        raise ValueError("q_n_text must contain only strings")
    return "\n".join([center] + neighbors)


class BM25Index:
    """Sparse postings index; no model, gold labels, or company edges required."""

    def __init__(self, documents, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be > 0 and b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.documents = {}
        self.code_to_id = {}
        self.lengths = {}
        self.postings = defaultdict(list)
        for row in documents:
            doc_id = row.get("doc_id")
            if not isinstance(doc_id, int) or isinstance(doc_id, bool):
                raise ValueError("document doc_id must be an integer")
            if doc_id in self.documents:
                raise ValueError("duplicate document ID {}".format(doc_id))
            for field in ("company_code", "company_name", "contents"):
                if not isinstance(row.get(field), str):
                    raise ValueError("document {}: {} must be a string".format(doc_id, field))
            code = row["company_code"]
            if not code or code in self.code_to_id:
                raise ValueError("empty or duplicate company code {!r}".format(code))
            self.documents[doc_id] = row
            self.code_to_id[code] = doc_id
            counts = Counter(tokenize(row["contents"]))
            self.lengths[doc_id] = sum(counts.values())
            for term, frequency in counts.items():
                self.postings[term].append((doc_id, frequency))
        if not self.documents:
            raise ValueError("no candidate documents")
        self.doc_ids = sorted(self.documents)
        self.doc_id_set = set(self.doc_ids)
        self.average_length = sum(self.lengths.values()) / float(len(self.documents))
        self.average_length = self.average_length or 1.0
        self.idf = {
            term: math.log(1.0 + (len(self.documents) - len(posting) + 0.5)
                           / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }
        self.length_norm = {
            doc_id: k1 * (1 - b + b * length / self.average_length)
            for doc_id, length in self.lengths.items()
        }

    def score(self, text):
        # Each unique query term contributes once: repeated portfolio phrases
        # cannot dominate just because the company owns more patent records.
        scores = defaultdict(float)
        for term in sorted(set(tokenize(text))):
            weight = self.idf.get(term)
            if weight is None:
                continue
            for doc_id, frequency in self.postings[term]:
                scores[doc_id] += weight * frequency * (self.k1 + 1) / (
                    frequency + self.length_norm[doc_id])
        return scores

    def rank(self, text, excluded=(), top_k=None):
        excluded = set(excluded)
        scores = self.score(text)
        candidates = (doc_id for doc_id in self.doc_ids if doc_id not in excluded)
        key = lambda doc_id: (-scores.get(doc_id, 0.0), doc_id)
        if top_k is None:
            ranked_ids = sorted(candidates, key=key)
        else:
            if top_k < 0:
                raise ValueError("top_k must be nonnegative")
            ranked_ids = heapq.nsmallest(top_k, candidates, key=key)
        return [(doc_id, scores.get(doc_id, 0.0)) for doc_id in ranked_ids]

    def recommendations(self, ranking):
        return [
            {
                "rank": rank,
                "doc_id": doc_id,
                "company_code": self.documents[doc_id]["company_code"],
                "company_name": self.documents[doc_id]["company_name"],
                "score": score,
            }
            for rank, (doc_id, score) in enumerate(ranking, 1)
        ]


def per_query_metrics(ranked_ids, positive_ids, cutoffs=RECALL_CUTOFFS):
    positives = set(positive_ids)
    if not positives:
        raise ValueError("query must have at least one positive")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranking contains duplicate candidate IDs")
    positive_ranks = [rank for rank, doc_id in enumerate(ranked_ids, 1)
                      if doc_id in positives]
    values = {
        "recall@{}".format(k): sum(rank <= k for rank in positive_ranks) / len(positives)
        for k in cutoffs
    }
    values["MRR"] = 1.0 / positive_ranks[0] if positive_ranks else 0.0
    values["P@1"] = float(bool(positive_ranks and positive_ranks[0] == 1))
    return values


def excluded_ids(query):
    """Always exclude the query's own company, even for a malformed export."""
    values = query.get("excluded_doc_ids", [])
    if not isinstance(values, list) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("excluded_doc_ids must be a list of integers")
    result = set(values)
    source_id = query.get("source_doc_id")
    if not isinstance(source_id, int) or isinstance(source_id, bool):
        raise ValueError("source_doc_id must be an integer")
    result.add(source_id)
    return result


def validate_query(index, query, require_positives=True):
    if not isinstance(query.get("query_id"), str) or not query["query_id"]:
        raise ValueError("query_id must be a nonempty string")
    if query.get("source_doc_id") not in index.doc_id_set:
        raise ValueError("{}: source is missing from candidates".format(query["query_id"]))
    query_text(query)
    excluded = excluded_ids(query)
    if not require_positives:
        return excluded, set()
    positive_list = query.get("positive_doc_ids")
    if not isinstance(positive_list, list) or not positive_list or any(
            not isinstance(value, int) or isinstance(value, bool) for value in positive_list):
        raise ValueError("{}: positive_doc_ids must be a nonempty list of integers".format(
            query["query_id"]))
    positives = set(positive_list)
    missing = positives - index.doc_id_set
    overlap = positives & excluded
    if missing or overlap:
        raise ValueError("{}: invalid positives: missing={}, excluded={}".format(
            query["query_id"], sorted(missing), sorted(overlap)))
    return excluded, positives


def evaluation_output_paths(data_dir, split, output):
    target = Path(output) if output else Path(data_dir) / "baseline"
    if target.suffix.lower() == ".jsonl":
        predictions = target
        metrics = target.with_name(target.stem + ".metrics.json")
    elif target.suffix.lower() == ".json":
        metrics = target
        predictions = target.with_name(target.stem + ".recommendations.jsonl")
    else:
        predictions = target / "recommendations.{}.jsonl".format(split)
        metrics = target / "metrics.{}.json".format(split)
    return predictions, metrics


def parameters(k1, b):
    return {
        "algorithm": "BM25", "tokenizer": TOKENIZER_VERSION,
        "k1": k1, "b": b, "query_term_frequency": "unique",
        "query_fields": ["q_text", "q_n_text"],
        "candidate_fields": ["contents"],
        "tie_breaker": "ascending_doc_id", "requires_gpu": False,
    }


def evaluate(data_dir, split="test", top_k=20, output=None, k1=1.5, b=0.75):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    start = time.monotonic()
    data_dir = Path(data_dir)
    document_path = data_dir / "documents.{}.jsonl".format(split)
    query_path = data_dir / "queries.{}.jsonl".format(split)
    predictions_path, metrics_path = evaluation_output_paths(data_dir, split, output)
    protected = {document_path.resolve(), query_path.resolve()}
    if predictions_path.resolve() in protected or metrics_path.resolve() in protected:
        raise ValueError("output path must not overwrite source data")
    fingerprints = {
        document_path.name: sha256_file(document_path),
        query_path.name: sha256_file(query_path),
    }
    for manifest_name in ("dataset_manifest.json", "manifest.json"):
        manifest_path = data_dir / manifest_name
        if manifest_path.exists():
            fingerprints[manifest_path.name] = sha256_file(manifest_path)
    index = BM25Index(read_jsonl(document_path), k1=k1, b=b)
    totals = {"recall@{}".format(k): 0.0 for k in RECALL_CUTOFFS}
    totals.update({"MRR": 0.0, "P@1": 0.0})
    seen_query_ids = set()
    query_count = 0
    total_positives = 0
    total_allowed_candidates = 0
    total_returned = 0
    zero_overlap_queries = 0
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # Gold partners are consulted after ranking; no candidate sampling occurs.
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for query in read_jsonl(query_path):
            excluded, positives = validate_query(index, query)
            qid = query["query_id"]
            if qid in seen_query_ids:
                raise ValueError("duplicate query ID {}".format(qid))
            seen_query_ids.add(qid)
            ranking = index.rank(query_text(query), excluded)
            values = per_query_metrics([doc_id for doc_id, _ in ranking], positives)
            for metric, value in values.items():
                totals[metric] += value
            query_count += 1
            total_positives += len(positives)
            total_allowed_candidates += len(ranking)
            shown = ranking[:top_k]
            total_returned += len(shown)
            zero_overlap_queries += int(not ranking or ranking[0][1] == 0)
            row = {
                "query_id": qid, "source_doc_id": query["source_doc_id"],
                "source_code": query.get("source_code", ""),
                "source_name": query.get("source_name", ""),
                "allowed_candidate_count": len(ranking),
                "positive_doc_ids": sorted(positives),
                "metrics": values,
                "recommendations": index.recommendations(shown),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not query_count:
        raise ValueError("no evaluation queries in {}".format(query_path))
    metrics = {metric: value / query_count for metric, value in totals.items()}
    metrics.update({
        "split": split, "query_count": query_count,
        "candidate_count": len(index.documents), "positive_count": total_positives,
        "mean_allowed_candidate_count": total_allowed_candidates / query_count,
        "mean_returned_candidate_count": total_returned / query_count,
        "zero_token_overlap_query_count": zero_overlap_queries,
        "recommendation_top_k": top_k, "evaluation_candidate_pool": "all_allowed_documents",
        "recall_definition": "macro_mean(hits_at_k / distinct_gold_partners_per_query)",
        "MRR_definition": "macro_mean(1 / first_gold_rank_in_entire_allowed_pool)",
        "P@1_definition": "fraction_of_queries_with_a_gold_partner_at_rank_1",
        "parameters": parameters(k1, b), "input_sha256": fingerprints,
        "recommendations_file": str(predictions_path.resolve()),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "limitations": [
            "Measures retrieval of observed first co-applicants, not confirmed future willingness.",
            "Unobserved company pairs are unlabeled, not verified nonpartners.",
            "This text-only baseline has no newly trained GraphFormers model.",
        ],
    })
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return metrics


def recommend(data_dir, split, company, top_k=5, k1=1.5, b=0.75):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    data_dir = Path(data_dir)
    document_path = data_dir / "documents.{}.jsonl".format(split)
    query_path = data_dir / "queries.{}.jsonl".format(split)
    index = BM25Index(read_jsonl(document_path), k1=k1, b=b)
    if company in index.code_to_id:
        source_id = index.code_to_id[company]
    else:
        matches = [doc_id for doc_id, row in index.documents.items()
                   if row["company_name"] == company]
        if not matches:
            raise ValueError("company not found in {}: {!r}".format(document_path, company))
        if len(matches) > 1:
            raise ValueError("ambiguous company name; specify company code: {!r}".format(company))
        source_id = matches[0]
    source = index.documents[source_id]
    selected = None
    if query_path.exists():
        for row in read_jsonl(query_path):
            if row.get("source_doc_id") == source_id:
                if selected is not None:
                    raise ValueError("multiple query snapshots for {}; use one split snapshot".format(company))
                selected = row
    if selected is not None:
        excluded, _ = validate_query(index, selected, require_positives=False)
        text = query_text(selected)
        exclusion_status = "self_and_dataset_known_partners"
        input_source = "query_pre_cutoff_portfolio"
    else:
        # Every candidate can be served when its document includes a historical
        # exclusion snapshot. Older exports lacking it explicitly use self only.
        document_query = dict(source)
        document_query["source_doc_id"] = source_id
        excluded = excluded_ids(document_query)
        exclusion_status = ("self_and_document_known_partners"
                            if "excluded_doc_ids" in source
                            else "self_only_no_history_export_for_company")
        if "q_text" in source or "q_n_text" in source:
            text = query_text(document_query)
            input_source = "document_query_pre_cutoff_portfolio"
        else:
            text = source["contents"]
            input_source = "document_pre_cutoff_portfolio"
    ranking = index.rank(text, excluded, top_k)
    return {
        "split": split, "source_doc_id": source_id,
        "source_code": source["company_code"], "source_name": source["company_name"],
        "input_source": input_source, "exclusion_status": exclusion_status,
        "excluded_doc_ids": sorted(excluded),
        "parameters": parameters(k1, b),
        "recommendations": index.recommendations(ranking),
        "meaning": "Potential technical partners ranked by patent text overlap; not verified willingness.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "recommend"):
        command = commands.add_parser(name)
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--split", choices=("train", "val", "test"), default="test")
        command.add_argument("--top-k", type=int, default=20 if name == "evaluate" else 5)
        command.add_argument("--k1", type=float, default=1.5)
        command.add_argument("--b", type=float, default=0.75)
        command.add_argument("--output", type=Path)
        if name == "recommend":
            command.add_argument("--company", required=True, help="Exact company name or company code")
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            result = evaluate(args.data_dir, args.split, args.top_k, args.output, args.k1, args.b)
        else:
            result = recommend(args.data_dir, args.split, args.company, args.top_k, args.k1, args.b)
            if args.output:
                protected = {args.data_dir / "documents.{}.jsonl".format(args.split),
                             args.data_dir / "queries.{}.jsonl".format(args.split)}
                if args.output.resolve() in {path.resolve() for path in protected}:
                    raise ValueError("output path must not overwrite source data")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
