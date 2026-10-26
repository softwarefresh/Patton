"""FTS5 lexical retrieval for patent-to-partner experiments (Python 3.8+).

Only dated document text is indexed. Answers are consulted after retrieval.
FTS5 BM25 and a bounded low-document-frequency query differ from the legacy
Python BM25 baseline, so their scores and metrics must not be conflated.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time

try:
    from .cooperation_baseline import (TOKENIZER_VERSION, evaluation_output_paths,
                                       query_text, read_jsonl, sha256_file, tokenize)
except ImportError:
    from cooperation_baseline import (TOKENIZER_VERSION, evaluation_output_paths,
                                      query_text, read_jsonl, sha256_file, tokenize)


OWNER = "patton_patent_partner_fts5_v1"
CUTOFFS = (5, 10, 20, 50, 100)


class FTSIndex:
    """Persistent contentless FTS index with stable IDs and zero-score padding."""

    def __init__(self, path, documents, fingerprint, max_query_terms=128):
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("fingerprint must be a nonempty string")
        if (not isinstance(max_query_terms, int) or isinstance(max_query_terms, bool)
                or max_query_terms < 1):
            raise ValueError("max_query_terms must be a positive integer")
        self.path = Path(path).resolve()
        self.fingerprint = fingerprint
        self.max_query_terms = max_query_terms
        self.documents = {}
        for row in documents:
            doc_id = row.get("doc_id")
            if (not isinstance(doc_id, int) or isinstance(doc_id, bool)
                    or not -(2 ** 63) <= doc_id < 2 ** 63):
                raise ValueError("doc_id must be a SQLite-range integer")
            if doc_id in self.documents:
                raise ValueError("duplicate doc_id {}".format(doc_id))
            if not isinstance(row.get("contents"), str):
                raise ValueError("document contents must be a string")
            self.documents[doc_id] = row
        if not self.documents:
            raise ValueError("no candidate documents")
        self.doc_ids = sorted(self.documents)
        self.doc_id_set = set(self.doc_ids)
        digest = hashlib.sha256()
        for doc_id in self.doc_ids:
            digest.update(json.dumps([doc_id, self.documents[doc_id]["contents"]],
                                     ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
        self.signature = {
            "owner": OWNER, "fingerprint": fingerprint,
            "document_digest": digest.hexdigest(), "tokenizer": TOKENIZER_VERSION,
            "document_count": str(len(self.doc_ids)), "index_schema_version": "2",
        }
        self.conn = None
        self.last_query_terms = ()
        self.last_query_token_count = 0
        self._df_cache = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_hit = self._cache_valid()
        if not self.cache_hit:
            self._build()
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("CREATE TEMP TABLE query_exclusions(doc_id INTEGER PRIMARY KEY)")

    def _cache_valid(self):
        if not self.path.exists():
            return False
        if not self.path.is_file():
            raise ValueError("index path is not a file")
        connection = sqlite3.connect(str(self.path))
        try:
            try:
                metadata = dict(connection.execute("SELECT key,value FROM index_meta"))
            except sqlite3.DatabaseError as exc:
                raise ValueError("refusing to overwrite an unrecognized index file: {}".format(self.path)) from exc
            if metadata.get("owner") != OWNER:
                raise ValueError("refusing to overwrite a foreign database: {}".format(self.path))
            if any(metadata.get(key) != value for key, value in self.signature.items()):
                return False
            try:
                ids = [row[0] for row in connection.execute("SELECT rowid FROM fts ORDER BY rowid")]
                return ids == self.doc_ids
            except sqlite3.DatabaseError:
                return False
        finally:
            connection.close()

    def _build(self):
        handle, temporary_name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".building", dir=str(self.path.parent))
        os.close(handle)
        temporary = Path(temporary_name)
        connection = None
        try:
            connection = sqlite3.connect(str(temporary))
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE index_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            connection.execute("CREATE VIRTUAL TABLE fts USING fts5(tokens,content='',tokenize='unicode61 remove_diacritics 0')")
            connection.execute("CREATE VIRTUAL TABLE vocab USING fts5vocab(fts,'row')")
            connection.execute("CREATE VIRTUAL TABLE vocab_instances USING fts5vocab(fts,'instance')")
            with connection:
                connection.executemany("INSERT INTO fts(rowid,tokens) VALUES (?,?)", (
                    (doc_id, " ".join(tokenize(self.documents[doc_id]["contents"])))
                    for doc_id in self.doc_ids))
                connection.executemany("INSERT INTO index_meta(key,value) VALUES (?,?)", self.signature.items())
                connection.execute("INSERT INTO fts(fts) VALUES ('optimize')")
            connection.close()
            connection = None
            temporary.replace(self.path)
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower():
                raise RuntimeError("SQLite FTS5 is required for this baseline") from exc
            raise
        finally:
            if connection is not None:
                connection.close()
            if temporary.exists():
                temporary.unlink()

    def _select_terms(self, text, excluded_count=0):
        if not isinstance(text, str):
            raise ValueError("query text must be a string")
        terms = sorted(set(tokenize(text)))
        self.last_query_token_count = len(terms)
        unknown = [term for term in terms if term not in self._df_cache]
        # Keep each statement below the older SQLite 999-variable limit.
        for offset in range(0, len(unknown), 512):
            batch = unknown[offset:offset + 512]
            found = dict(self.conn.execute(
                "SELECT term,doc FROM vocab WHERE term IN ({})".format(",".join("?" for _ in batch)), batch))
            for term in batch:
                self._df_cache[term] = int(found.get(term, 0))
        usable = [term for term in terms if self._df_cache[term] > 0]
        usable.sort(key=lambda term: (self._df_cache[term], term))
        selected = []
        for offset in range(0, len(usable), 256):
            batch = usable[offset:offset + 256]
            # A term occurring in more documents than the entire exclusion set
            # must occur in an allowed document. Only rarer terms need postings.
            available = {term for term in batch if self._df_cache[term] > excluded_count}
            uncertain = [term for term in batch if term not in available]
            if uncertain:
                available.update(row[0] for row in self.conn.execute(
                    "SELECT v.term FROM vocab AS v WHERE v.term IN ({}) "
                    "AND EXISTS(SELECT 1 FROM vocab_instances AS i "
                    "WHERE i.term=v.term AND i.doc NOT IN "
                    "(SELECT doc_id FROM query_exclusions))".format(",".join("?" for _ in uncertain)), uncertain))
            selected.extend(term for term in batch if term in available)
            if len(selected) >= self.max_query_terms:
                break
        self.last_query_terms = tuple(selected[:self.max_query_terms])
        return self.last_query_terms

    def rank(self, text, excluded=(), top_k=100):
        if self.conn is None:
            raise RuntimeError("FTSIndex is closed")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0):
            raise ValueError("top_k must be a nonnegative integer or None")
        excluded = set(excluded)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in excluded):
            raise ValueError("excluded IDs must be integers")
        excluded &= self.doc_id_set
        limit = len(self.doc_ids) - len(excluded)
        if top_k is not None:
            limit = min(limit, top_k)
        if limit <= 0:
            self.last_query_terms = ()
            self.last_query_token_count = 0
            return []
        with self.conn:
            self.conn.execute("DELETE FROM query_exclusions")
            self.conn.executemany("INSERT INTO query_exclusions(doc_id) VALUES (?)", ((value,) for value in excluded))
        terms = self._select_terms(text, len(excluded))
        ranking = []
        if terms:
            match_query = " OR ".join('"{}"'.format(term.replace('"', '""')) for term in terms)
            rows = self.conn.execute(
                "SELECT rowid,bm25(fts) FROM fts WHERE fts MATCH ? "
                "AND rowid NOT IN (SELECT doc_id FROM query_exclusions) "
                "ORDER BY bm25(fts),rowid LIMIT ?", (match_query, limit))
            for doc_id, raw_score in rows:
                score = max(0.0, -float(raw_score))
                if not math.isfinite(score):
                    raise ValueError("non-finite FTS score")
                ranking.append((int(doc_id), score))
        if len(ranking) < limit:
            used = {doc_id for doc_id, _ in ranking}
            for doc_id in self.doc_ids:
                if doc_id not in excluded and doc_id not in used:
                    ranking.append((doc_id, 0.0))
                    if len(ranking) == limit:
                        break
        return ranking

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def per_query_metrics(ranking, positive_ids):
    """Zero-score padding is not a lexical retrieval hit, even if ID-sorted."""
    positives = set(positive_ids)
    if not positives:
        raise ValueError("query must have at least one positive")
    ids = [doc_id for doc_id, _ in ranking]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate IDs in ranking")
    hits = [rank for rank, (doc_id, score) in enumerate(ranking, 1)
            if doc_id in positives and score > 0]
    result = {}
    for cutoff in CUTOFFS:
        count = sum(rank <= cutoff for rank in hits)
        result["recall@{}".format(cutoff)] = count / len(positives)
        result["Hit@{}".format(cutoff)] = float(count > 0)
    result["P@1"] = float(bool(hits and hits[0] == 1))
    result["MRR@100"] = 1.0 / hits[0] if hits and hits[0] <= 100 else 0.0
    return result


def _query_protocol(query):
    if not isinstance(query.get("query_id"), str) or not query["query_id"]:
        raise ValueError("query_id must be a nonempty string")
    excluded = query.get("excluded_doc_ids", [])
    positives = query.get("positive_doc_ids", [])
    for name, values in (("excluded_doc_ids", excluded), ("positive_doc_ids", positives)):
        if not isinstance(values, list) or any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError(name + " must be a list of integers")
    excluded, positives = set(excluded), set(positives)
    source = query.get("source_doc_id")
    if not isinstance(source, int) or isinstance(source, bool):
        raise ValueError("source_doc_id must be an integer")
    excluded.add(source)
    if not positives or positives & excluded:
        raise ValueError("positives must be nonempty and not excluded")
    return excluded, positives


def evaluate(data_dir, split="test", output=None, max_query_terms=128, top_k=100):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    started = time.monotonic()
    data_dir = Path(data_dir).resolve()
    document_path = data_dir / "documents.{}.jsonl".format(split)
    query_path = data_dir / "queries.{}.jsonl".format(split)
    predictions_path, metrics_path = evaluation_output_paths(data_dir, split, output)
    protected = {document_path.resolve(), query_path.resolve()}
    fingerprints = {path.name: sha256_file(path) for path in (document_path, query_path)}
    for name in ("dataset_manifest.json", "manifest.json"):
        path = data_dir / name
        if path.exists():
            fingerprints[name] = sha256_file(path)
            protected.add(path.resolve())
    if predictions_path.resolve() in protected or metrics_path.resolve() in protected:
        raise ValueError("output must not overwrite source data")
    documents = list(read_jsonl(document_path))
    prepared_index = data_dir / "bm25.{}.sqlite".format(split)
    index_path = (prepared_index if prepared_index.is_file() else
                  data_dir / "fts_cache" / "documents.{}.sqlite3".format(split))
    totals = {"recall@{}".format(k): 0.0 for k in CUTOFFS}
    totals.update({"Hit@{}".format(k): 0.0 for k in CUTOFFS})
    totals.update({"P@1": 0.0, "MRR@100": 0.0})
    counts = {"queries": 0, "positive_occurrences": 0, "allowed_candidates": 0,
              "returned_candidates": 0, "zero_overlap_queries": 0,
              "gold_outside_pool_occurrences": 0, "queries_with_gold_outside_pool": 0,
              "selected_terms": 0}
    seen = set()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with FTSIndex(index_path, documents, fingerprints[document_path.name], max_query_terms) as index:
        cache_hit = index.cache_hit
        with predictions_path.open("w", encoding="utf-8", newline="\n") as sink:
            for query in read_jsonl(query_path):
                qid = query.get("query_id")
                if qid in seen:
                    raise ValueError("duplicate query ID {}".format(qid))
                excluded, positives = _query_protocol(query)
                seen.add(qid)
                # Ranking receives text and exclusions only, never positives.
                ranking = index.rank(query_text(query), excluded, max(100, top_k))
                values = per_query_metrics(ranking, positives)
                for name, value in values.items():
                    totals[name] += value
                missing = positives - index.doc_id_set
                counts["queries"] += 1
                counts["positive_occurrences"] += len(positives)
                counts["allowed_candidates"] += len(index.doc_ids) - len(excluded & index.doc_id_set)
                counts["returned_candidates"] += len(ranking[:top_k])
                counts["zero_overlap_queries"] += int(not ranking or ranking[0][1] <= 0)
                counts["gold_outside_pool_occurrences"] += len(missing)
                counts["queries_with_gold_outside_pool"] += int(bool(missing))
                counts["selected_terms"] += len(index.last_query_terms)
                sink.write(json.dumps({"query_id": qid, "source_doc_id": query["source_doc_id"],
                    "positive_doc_ids": sorted(positives), "metrics": values,
                    "allowed_candidate_count": len(index.doc_ids) - len(excluded & index.doc_id_set),
                    "selected_query_term_count": len(index.last_query_terms),
                    "recommendations": [{"rank": rank, "doc_id": doc_id, "score": score,
                        "company_code": index.documents[doc_id].get("company_code", ""),
                        "company_name": index.documents[doc_id].get("company_name", "")}
                        for rank, (doc_id, score) in enumerate(ranking[:top_k], 1)]}, ensure_ascii=False) + "\n")
    n = counts["queries"]
    if not n:
        raise ValueError("no evaluation queries")
    metrics = {name: value / n for name, value in totals.items()}
    metrics.update({"split": split, "query_count": n, "candidate_count": len(documents),
        "positive_count": counts["positive_occurrences"],
        "mean_allowed_candidate_count": counts["allowed_candidates"] / n,
        "mean_returned_candidate_count": counts["returned_candidates"] / n,
        "zero_token_overlap_query_count": counts["zero_overlap_queries"],
        "gold_outside_pool_occurrences": counts["gold_outside_pool_occurrences"],
        "queries_with_gold_outside_pool": counts["queries_with_gold_outside_pool"],
        "mean_selected_query_terms": counts["selected_terms"] / n,
        "recommendation_top_k": top_k, "metric_depth": 100,
        "evaluation_candidate_pool": "all_allowed_documents",
        "recall_definition": "macro_mean(positive_score_hits_at_k / all_distinct_gold_partners)",
        "hit_definition": "fraction_of_queries_with_a_positive_score_gold_at_k",
        "MRR_definition": "first_positive_score_gold_reciprocal_rank_at_100; missing=0",
        "zero_score_padding_counts_as_retrieval_hit": False,
        "parameters": {"algorithm": "SQLite FTS5 built-in BM25", "tokenizer": TOKENIZER_VERSION,
            "sqlite_version": sqlite3.sqlite_version, "max_query_terms": max_query_terms,
            "query_term_selection": "terms_matching_an_allowed_candidate_by_global_document_frequency_then_term",
            "document_frequency_source": "fts5vocab(row)", "match_operator": "quoted_terms_OR",
            "term_eligibility_source": "batched_fts5vocab(instance)_existence_after_exclusions",
            "candidate_exclusions_affect_term_eligibility": True,
            "evaluation_exclusions": ["self", "exported_prior_partners_and_group_risks"],
            "gold_used_for_scoring": False,
            "query_fields": ["q_text", "q_n_text"], "candidate_fields": ["contents"],
            "tie_breaker": "ascending_doc_id", "zero_score_padding": True,
            "requires_gpu": False, "index_cache_hit": cache_hit},
        "input_sha256": fingerprints, "index_file": str(index_path),
        "recommendations_file": str(predictions_path.resolve()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": ["FTS5 BM25 and bounded low-df terms differ from the legacy Python BM25 baseline.",
            "Zero-score padded candidates are reported but do not receive retrieval metric credit.",
            "Hard-negative callers may exclude known positives for sampling; evaluation never adds gold to exclusions.",
            "Unobserved candidate pairs are unlabeled, not verified nonpartners.",
            "This is a lexical baseline, not a newly trained GraphFormers model."]})
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("evaluate")
    command.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/patent/cooperation_v2")
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument("--output", type=Path)
    command.add_argument("--max-query-terms", type=int, default=128)
    command.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    metrics = evaluate(args.data_dir, args.split, args.output, args.max_query_terms, args.top_k)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
