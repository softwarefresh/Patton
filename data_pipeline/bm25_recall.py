"""BM25 全库 recall@K 评测（本地 CPU）: test 查询 vs 25,598 家候选池。

与 stage_bm25 完全相同的口径（bigram 分词 + MIN_DF=5 词表过滤 + 相同 BM25 参数），
唯一区别: 对 test 查询取 top-200 并对照 q_pos 算 recall@20/50/100/200。
W/Q 用 scipy 稀疏存储, 数值与稠密 BLAS 版完全一致, 内存从几十 GB 降到几十 MB。

用法: python bm25_recall.py
"""
import os
import sqlite3
from collections import Counter

import numpy as np
import scipy.sparse as sp

DP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DB = os.path.join(DP_DIR, 'build_cache', 'cache.db')

PUNCT = '，。；：、()（）-'


def _tokenize_bigram(text):
    toks = []
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        if a.isspace() or b.isspace() or a in PUNCT or b in PUNCT:
            continue
        toks.append(a + b)
    return toks or [text[:2]]


def main():
    con = sqlite3.connect(CACHE_DB)
    docs = con.execute('SELECT doc_id, k_text FROM doc_map ORDER BY doc_id').fetchall()
    doc_ids = [d[0] for d in docs]
    tokenized = [_tokenize_bigram(d[1]) for d in docs]
    N = len(docs)
    dl = np.array([len(t) for t in tokenized], dtype=np.float32)
    avgdl = float(dl.mean())

    vocab = {}
    for t in tokenized:
        for tok in t:
            vocab.setdefault(tok, len(vocab))
    V = len(vocab)
    print(f'BM25: 文档 {N}, 原始词表 {V}, 平均文档长度 {avgdl:.1f}', flush=True)

    MIN_DF = 5
    df_raw = np.zeros(V, dtype=np.int32)
    for t in tokenized:
        for tok in set(t):
            df_raw[vocab[tok]] += 1
    keep = df_raw >= MIN_DF
    V2 = int(keep.sum())
    old2new = np.full(V, -1, dtype=np.int32)
    old2new[keep] = np.arange(V2)
    del df_raw
    print(f'BM25: 过滤 df>={MIN_DF} 后词表 {V2}', flush=True)

    # 稀疏 tf -> W, 与 stage_bm25 数值逐项一致: w = f*2.5/(f+const_i) * idf[col]
    rows, cols, vals = [], [], []
    for i, t in enumerate(tokenized):
        for tok, f in Counter(t).items():
            v = vocab.get(tok, -1)
            nv = old2new[v] if v >= 0 else -1
            if nv >= 0:
                rows.append(i)
                cols.append(nv)
                vals.append(f)
    del tokenized
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    vals = np.array(vals, dtype=np.float32)
    df = np.zeros(V2, dtype=np.float32)
    # df = 出现该词的文档数; (row, col) 对本身唯一（每文档内词频已用 Counter 合并）, 直接按列累加
    uniq = np.ones(len(rows), dtype=bool)
    uniq[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    np.add.at(df, cols[uniq], 1)
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    const = (1.5 * (1 - 0.75 + 0.75 * (dl / avgdl))).astype(np.float32)
    vals = vals * 2.5 / (vals + const[rows]) * idf[cols]
    W = sp.coo_matrix((vals, (rows, cols)), shape=(N, V2)).tocsr()
    print(f'W: {W.shape}, nnz {W.nnz}, 内存 {W.data.nbytes/1e6:.0f}MB', flush=True)

    queries = con.execute("SELECT qid, q_text FROM queries WHERE split='test' ORDER BY qid").fetchall()
    pos_map = {}
    for qid, did in con.execute('SELECT qid, doc_id FROM q_pos'):
        pos_map.setdefault(qid, set()).add(did)
    doc_id_set = set(doc_ids)
    n_missing = sum(1 for qid, _ in queries
                    if any(d not in doc_id_set for d in pos_map.get(qid, [])))
    print(f'test 查询 {len(queries)}; 正例不在 doc_map 的查询数: {n_missing}', flush=True)

    K = 200
    KS = (20, 50, 100, 200)
    hits = {k: 0.0 for k in KS}
    n = 0
    chunk = 2000
    for start in range(0, len(queries), chunk):
        qs = queries[start:start + chunk]
        q_rows, q_cols, q_vals = [], [], []
        for qi, (_, qt) in enumerate(qs):
            for tok, f in Counter(_tokenize_bigram(qt)).items():
                v = vocab.get(tok, -1)
                nv = old2new[v] if v >= 0 else -1
                if nv >= 0:
                    q_rows.append(qi)
                    q_cols.append(nv)
                    q_vals.append(f)
        Q = sp.coo_matrix((np.array(q_vals, dtype=np.float32),
                           (np.array(q_rows, dtype=np.int32), np.array(q_cols, dtype=np.int32))),
                          shape=(len(qs), V2)).tocsr()
        scores = (Q @ W.T).toarray()  # (chunk, N)
        del Q
        order = np.argpartition(-scores, K - 1, axis=1)[:, :K]
        order = order[np.arange(len(qs))[:, None],
                      np.argsort(-scores[np.arange(len(qs))[:, None], order], axis=1)]
        del scores
        for qi, (qid, _) in enumerate(qs):
            pos = pos_map.get(qid)
            if not pos:
                continue
            n += 1
            top = [doc_ids[int(d)] for d in order[qi]]
            for k in KS:
                hits[k] += sum(1 for d in top[:k] if d in pos) / len(pos)
        del order
        print(f'{min(start + chunk, len(queries))}/{len(queries)}  done', flush=True)

    con.close()
    print('---')
    for k in KS:
        print(f'recall@{k} = {hits[k] / n:.4f}  (n={n})')


if __name__ == '__main__':
    main()
