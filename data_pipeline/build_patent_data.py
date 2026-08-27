"""专利侧训练输入构建主脚本（分阶段，可单独重跑）。

用法:
  python build_patent_data.py documents   # k_text + doc id 映射 + documents.json/txt
  python build_patent_data.py queries     # 查询集(2022-2024 正例在池) + 时间切分 + 正例
  python build_patent_data.py neighbors   # q_n(同IPC×5, 查询+预训练) + k_n(专利组合+合作方×5)
  python build_patent_data.py bm25        # BM25 硬负例(每查询 top20, 过滤正例)
  python build_patent_data.py assemble    # 组装 data/patent/{nc,pretrain} 全部文本文件

中间产物: data_pipeline/build_cache/cache.db
  doc_map(doc_id,code,name,type,k_text) / queries(qid,申请号,year,split,q_text,ipc_main)
  q_pos(qid,doc_id) / q_neighbors(qid,n_idx,申请号) / pretrain_qn(申请号,n_idx,neighbor)
  k_neighbors(doc_id,n_idx,kind,ref) / bm25_neg(qid,rank,doc_id)
最终产物: data/patent/nc/ 与 data/patent/pretrain/（格式对齐原版 Patton, 见 README 与 build_train*.py）

关键设计:
  - 时间切分: 2022=train, 2023=val, 2024=test（时序评估, 防泄漏）
  - 预训练对: 申请年份<=2023 的全部池内链接对（排除测试年）
  - 本地截断: q[:400] qn[:300] k[:500] kn[:300] neg_k[:300]（tokenize 时 max_len=256 反正截断）
  - 检索负例 8 个(4 BM25硬+4随机, k_n 留空); 重排负例 20 个纯 BM25(更硬), 训练只取 10 万查询样本
  - 邻居槽数 N=5, 空槽用 ''（模型 mask=0 处理）
"""
import argparse
import csv
import json
import os
import random
import sqlite3
import sys
from collections import Counter, defaultdict

import numpy as np

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP_DIR = os.path.join(PROJ_DIR, 'data_pipeline')
CACHE_DIR = os.path.join(DP_DIR, 'build_cache')
CACHE_DB = os.path.join(CACHE_DIR, 'cache.db')
DB = os.path.join(DP_DIR, 'g_domain.db')
INFO = os.path.join(DP_DIR, 'g_company_info.csv')
OUT_NC = os.path.join(PROJ_DIR, 'data', 'patent', 'nc')
OUT_PRE = os.path.join(PROJ_DIR, 'data', 'patent', 'pretrain')

N_NB = 5                 # 邻居槽数
K_PORTFOLIO = 3          # k_n 里企业专利数
K_PARTNER = 2            # k_n 里合作方数
TRUNC_Q = 400            # 本地截断: 查询文本
TRUNC_N = 300            # 邻居文本
TRUNC_K = 500            # 正例企业文本
TRUNC_NEG = 300          # 负例企业文本
# ⚠️ 上面两个值不相等 = 正负例第二处构造不对称(正例最长 500 字、负例 300 字)。
# 当前 max_len=256, 中文近似字级 tokenize, 两者都会被 tokenizer 截到 ~254 token,
# 短于 300 字的文本两侧完全一致 —— 所以现在观察不到差异、不构成泄漏。
# 但 max_len 一旦提到 300 以上, "文本更长=正例" 就会变成可利用的捷径。
# 若要提高 max_len, 先把这两个值改成相等。

N_BM25_HARD = 4          # 检索负例中 BM25 硬负个数
N_RAND = 4               # 检索负例中随机负个数
N_NEGS_RERANK = 20       # 重排每查询 BM25 硬负个数
RERANK_TRAIN_SAMPLE = 100000   # 重排训练采样查询数


def get_cache():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return sqlite3.connect(CACHE_DB)


# ---------------------------------------------------------------- documents
def stage_documents():
    con = get_cache()
    con.execute('DROP TABLE IF EXISTS doc_map')
    con.execute('CREATE TABLE doc_map (doc_id INTEGER PRIMARY KEY, code TEXT UNIQUE, name TEXT, type TEXT, k_text TEXT)')
    with open(INFO, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))[1:]
    docs = []
    for i, r in enumerate(rows):
        code, name, typ, _, scope, industry, _, _, _ = r
        parts = [name]
        if scope and scope not in ('-', '未公示'):
            parts.append('经营范围：' + scope)
        if industry and industry != '-':
            parts.append('所属行业：' + industry)
        k_text = '。'.join(parts)
        con.execute('INSERT INTO doc_map VALUES (?,?,?,?,?)', (i, code, name, typ, k_text))
        docs.append({'id': i, 'contents': k_text})
    con.commit()
    os.makedirs(OUT_NC, exist_ok=True)
    with open(os.path.join(OUT_NC, 'documents.json'), 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False)
    with open(os.path.join(OUT_NC, 'documents.txt'), 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(f"{d['id']}\t{d['contents']}\n")
    con.close()
    print(f'documents: {len(docs)} 家企业')
    print('k_text 样例:', docs[0]['contents'][:120], flush=True)


# ---------------------------------------------------------------- queries
def stage_queries():
    db = sqlite3.connect(DB)
    cache = get_cache()
    code2doc = dict(cache.execute('SELECT code, doc_id FROM doc_map').fetchall())
    codes = list(code2doc.keys())

    cur = db.cursor()
    # 池代码临时表(带索引), 避免 2.5 万项 IN 列表的 O(N^2) 计划
    cur.execute('CREATE TEMP TABLE pool(code TEXT PRIMARY KEY)')
    cur.executemany('INSERT INTO pool VALUES(?)', [(c,) for c in codes])
    # 2022-2024 池内链接 -> 专利(年份/q_text/ipc) 与 正例 doc_id
    qinfo = {}
    cur.execute('''SELECT l.申请号, p.申请年份, p.q_text, p.ipc_main, l.信用代码
        FROM links_g l
        JOIN pool ON pool.code = l.信用代码
        JOIN patents_g p ON p.申请号 = l.申请号
        WHERE p.申请年份 BETWEEN 2022 AND 2024''')
    for appn, year, qtext, ipc, code in cur:
        d = qinfo.setdefault(appn, {'year': year, 'q_text': qtext, 'ipc_main': ipc or '', 'pos': set()})
        d['pos'].add(code2doc[code])
    db.close()

    split = {'train': 2022, 'val': 2023, 'test': 2024}
    cache.execute('DROP TABLE IF EXISTS queries')
    cache.execute('DROP TABLE IF EXISTS q_pos')
    cache.execute('''CREATE TABLE queries (qid INTEGER PRIMARY KEY, 申请号 TEXT UNIQUE,
        year INTEGER, split TEXT, q_text TEXT, ipc_main TEXT)''')
    cache.execute('CREATE TABLE q_pos (qid INTEGER, doc_id INTEGER, PRIMARY KEY (qid, doc_id))')
    qid = 0
    cnt = Counter()
    for appn, d in sorted(qinfo.items()):
        if d['year'] not in split.values() or not d['pos']:
            continue
        s = [k for k, v in split.items() if v == d['year']][0]
        cache.execute('INSERT INTO queries VALUES (?,?,?,?,?,?)',
                      (qid, appn, d['year'], s, d['q_text'], d['ipc_main']))
        for did in d['pos']:
            cache.execute('INSERT INTO q_pos VALUES (?,?)', (qid, did))
        cnt[s] += 1
        qid += 1
    cache.commit()
    cache.close()
    print('查询集:', dict(cnt), '合计', qid)


# ---------------------------------------------------------------- neighbors
def _sample_ipc_neighbors(ipc_groups, self_appn, ipc, n, rng):
    """O(n) 采样: 抽 n+1 个再剔除自身(避免每次重建候选列表 O(组大小))。"""
    if not ipc:
        return []
    group = ipc_groups.get(ipc)
    if not group:
        return []
    picks = rng.sample(group, min(n + 1, len(group)))
    picks = [a for a in picks if a != self_appn]
    for a in group:
        if len(picks) >= n:
            break
        if a not in picks:
            picks.append(a)  # 补足
    return picks[:n]


def stage_neighbors():
    db = sqlite3.connect(DB)
    cache = get_cache()
    code2doc = dict(cache.execute('SELECT code, doc_id FROM doc_map').fetchall())
    doc2code = {v: k for k, v in code2doc.items()}
    codes = list(code2doc.keys())

    cur = db.cursor()
    cur.execute('CREATE TEMP TABLE pool(code TEXT PRIMARY KEY)')
    cur.executemany('INSERT INTO pool VALUES(?)', [(c,) for c in codes])
    # IPC 分组
    ipc_groups = defaultdict(list)
    for ipc, appn in cur.execute('SELECT ipc_main, 申请号 FROM patents_g WHERE ipc_main != ""'):
        ipc_groups[ipc].append(appn)
    print('IPC 组数:', len(ipc_groups))

    rng = random.Random(42)

    # ---- q 邻居(查询专利)
    cache.execute('DROP TABLE IF EXISTS q_neighbors')
    cache.execute('CREATE TABLE q_neighbors (qid INTEGER, n_idx INTEGER, 申请号 TEXT)')
    for qid, appn, ipc in cache.execute('SELECT qid, 申请号, ipc_main FROM queries').fetchall():
        for i, a in enumerate(_sample_ipc_neighbors(ipc_groups, appn, ipc, N_NB, rng)):
            cache.execute('INSERT INTO q_neighbors VALUES (?,?,?)', (qid, i, a))
    cache.commit()
    print('q_neighbors(查询) 完成')

    # ---- q 邻居(预训练专利: 2023 及以前, 池内链接)
    cache.execute('DROP TABLE IF EXISTS pretrain_qn')
    cache.execute('CREATE TABLE pretrain_qn (申请号 TEXT, n_idx INTEGER, neighbor TEXT)')
    cur.execute('''SELECT DISTINCT l.申请号, p.ipc_main FROM links_g l
        JOIN pool ON pool.code = l.信用代码
        JOIN patents_g p ON p.申请号 = l.申请号
        WHERE p.申请年份 <= 2023''')
    n_pre = 0
    for appn, ipc in cur:
        picks = _sample_ipc_neighbors(ipc_groups, appn, ipc or '', N_NB, rng)
        for i, a in enumerate(picks):
            cache.execute('INSERT INTO pretrain_qn VALUES (?,?,?)', (appn, i, a))
        n_pre += 1
        if n_pre % 200000 == 0:
            cache.commit()
            print(f'  pretrain_qn 进度: {n_pre}', flush=True)
    cache.commit()
    print(f'pretrain_qn 完成: {n_pre} 专利', flush=True)

    # ---- k 邻居(全部企业): 专利组合 3 + 合作方 2
    cache.execute('DROP TABLE IF EXISTS k_neighbors')
    cache.execute('CREATE TABLE k_neighbors (doc_id INTEGER, n_idx INTEGER, kind TEXT, ref TEXT)')
    # 池内企业名下专利(取前3, 按申请号排序保证稳定)
    cur.execute('''SELECT l.信用代码, l.申请号 FROM links_g l
        JOIN pool ON pool.code = l.信用代码 ORDER BY l.信用代码, l.申请号''')
    comp_patents = defaultdict(list)
    for code, appn in cur:
        if len(comp_patents[code]) < K_PORTFOLIO:
            comp_patents[code].append(appn)
    # 合作方: 同专利共现, 只统计池内对
    cur.execute('''SELECT l1.信用代码, l2.信用代码, COUNT(*) AS c FROM links_g l1
        JOIN links_g l2 ON l1.申请号 = l2.申请号
        WHERE l1.信用代码 != l2.信用代码 GROUP BY l1.信用代码, l2.信用代码''')
    partners = defaultdict(list)
    for c1, c2, cnt in cur:
        if c1 in code2doc and c2 in code2doc:
            partners[c1].append((c2, cnt))
    db.close()

    n_missing = 0
    for doc_id, code in doc2code.items():
        slots = [(i, 'patent', a) for i, a in enumerate(comp_patents.get(code, []))]
        part = sorted(partners.get(code, []), key=lambda x: -x[1])[:K_PARTNER]
        for j, (c2, _) in enumerate(part):
            slots.append((len(slots), 'doc', str(code2doc[c2])))
        for i, kind, ref in slots[:N_NB]:
            cache.execute('INSERT INTO k_neighbors VALUES (?,?,?,?)', (doc_id, i, kind, ref))
        if not slots:
            n_missing += 1
    cache.commit()
    cache.close()
    print(f'k_neighbors 完成, 无邻居企业数: {n_missing}', flush=True)


# ---------------------------------------------------------------- bm25
def _tokenize_bigram(text):
    toks = []
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        if a.isspace() or b.isspace() or a in '，。；：、()（）-' or b in '，。；：、()（）-':
            continue
        toks.append(a + b)
    return toks or [text[:2]]


def stage_bm25():
    cache = get_cache()
    docs = cache.execute('SELECT doc_id, k_text FROM doc_map ORDER BY doc_id').fetchall()
    doc_ids = [d[0] for d in docs]
    tokenized = [_tokenize_bigram(d[1]) for d in docs]
    N = len(docs)
    dl = np.array([len(t) for t in tokenized], dtype=np.float32)
    avgdl = dl.mean()

    vocab = {}
    for t in tokenized:
        for tok in t:
            vocab.setdefault(tok, len(vocab))
    V = len(vocab)
    print(f'BM25: 文档 {N}, 原始词表 {V}, 平均文档长度 {avgdl:.1f}', flush=True)

    # 词表裁剪: 只保留 df>=MIN_DF 的二元组, 控制稠密矩阵内存
    MIN_DF = 5
    df_raw = np.zeros(V, dtype=np.int32)
    for t in tokenized:
        for tok in set(t):
            df_raw[vocab[tok]] += 1
    keep = df_raw >= MIN_DF
    V2 = int(keep.sum())
    old2new = np.full(V, -1, dtype=np.int32)
    old2new[keep] = np.arange(V2)
    print(f'BM25: 过滤 df>={MIN_DF} 后词表 {V2}', flush=True)

    # 稠密 tf/idf/权重矩阵 (N, V2) — 用 BLAS 打分
    tf = np.zeros((N, V2), dtype=np.float32)
    for i, t in enumerate(tokenized):
        for tok, f in Counter(t).items():
            v = vocab.get(tok, -1)
            nv = old2new[v] if v >= 0 else -1
            if nv >= 0:
                tf[i, nv] = f
    df = (tf > 0).sum(axis=0).astype(np.float32)
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    const = (1.5 * (1 - 0.75 + 0.75 * (dl / avgdl)))[:, None]
    W = tf * 2.5 / (tf + const) * idf[None, :]
    del tf
    print('文档权重矩阵 W:', W.shape, '内存 %.1fGB' % (W.nbytes / 1e9), flush=True)

    cache.execute('DROP TABLE IF EXISTS bm25_neg')
    cache.execute('CREATE TABLE bm25_neg (qid INTEGER, rank INTEGER, doc_id INTEGER)')

    queries = cache.execute('SELECT qid, q_text FROM queries ORDER BY qid').fetchall()
    chunk = 5000
    n_done = 0
    for start in range(0, len(queries), chunk):
        qs = queries[start:start + chunk]
        Q = np.zeros((len(qs), V2), dtype=np.float32)
        for qi, (_, qt) in enumerate(qs):
            for tok, f in Counter(_tokenize_bigram(qt)).items():
                v = vocab.get(tok, -1)
                nv = old2new[v] if v >= 0 else -1
                if nv >= 0:
                    Q[qi, nv] = f
        scores = Q @ W.T  # (chunk, N) 稠密, BLAS
        order = np.argpartition(-scores, N_NEGS_RERANK - 1, axis=1)[:, :N_NEGS_RERANK]
        order = order[np.arange(len(qs))[:, None],
                      np.argsort(-scores[np.arange(len(qs))[:, None], order], axis=1)]
        for qi, (qid, _) in enumerate(qs):
            pos = set(r[0] for r in cache.execute('SELECT doc_id FROM q_pos WHERE qid=?', (qid,)).fetchall())
            rank = 0
            for d in order[qi]:
                did = doc_ids[int(d)]
                if did in pos:
                    continue
                cache.execute('INSERT INTO bm25_neg VALUES (?,?,?)', (qid, rank, did))
                rank += 1
                if rank >= N_NEGS_RERANK:
                    break
        cache.commit()
        n_done += len(qs)
        print(f'BM25 打分: {n_done}/{len(queries)}', flush=True)
    cache.close()
    print('bm25 负例完成', flush=True)


# ---------------------------------------------------------------- assemble
def _fetch_patent_texts(db_cur, appns, trunc):
    out = {}
    appns = list(appns)
    for i in range(0, len(appns), 5000):
        part = appns[i:i + 5000]
        db_cur.execute('SELECT 申请号, q_text FROM patents_g WHERE 申请号 IN (%s)' % ','.join('?' * len(part)), part)
        for a, t in db_cur:
            out[a] = (t or '')[:trunc]
    return out


def _load_company_kn(cache, db_cur, doc_ktext):
    """全部企业 k_n_text(5 槽) -> {doc_id: [text×5]}（专利文本[:300]/合作方 k_text[:300]）。"""
    need = set()
    refs = {}
    for did, n_idx, kind, ref in cache.execute('SELECT doc_id, n_idx, kind, ref FROM k_neighbors'):
        refs.setdefault(did, {})[n_idx] = (kind, ref)
        if kind == 'patent':
            need.add(ref)
    appn2t = _fetch_patent_texts(db_cur, need, TRUNC_N)
    kn = {}
    for did, slots in refs.items():
        out = [''] * N_NB
        for n_idx, (kind, ref) in slots.items():
            out[n_idx] = appn2t.get(ref, '') if kind == 'patent' else doc_ktext.get(int(ref), '')[:TRUNC_N]
        kn[did] = out
    return kn


def stage_assemble():
    db = sqlite3.connect(DB)
    cache = get_cache()
    # 组装按 qid/申请号 频繁点查, 必须建索引(否则全表扫描 480 万行)
    for idx_sql in ['CREATE INDEX IF NOT EXISTS idx_qn_qid ON q_neighbors(qid)',
                    'CREATE INDEX IF NOT EXISTS idx_pqn_appn ON pretrain_qn(申请号)',
                    'CREATE INDEX IF NOT EXISTS idx_bm_qid ON bm25_neg(qid)']:
        cache.execute(idx_sql)
    cache.commit()
    doc_ktext = {d: (t or '')[:TRUNC_NEG] for d, t in cache.execute('SELECT doc_id, k_text FROM doc_map')}
    doc_ktext_full = {d: (t or '')[:TRUNC_K] for d, t in cache.execute('SELECT doc_id, k_text FROM doc_map')}
    db_cur = db.cursor()
    print('构建企业 k_n_text ...', flush=True)
    kn = _load_company_kn(cache, db_cur, {d: (t or '')[:TRUNC_NEG] for d, t in doc_ktext.items()})

    # 查询邻居文本: 按 chunk 流式
    def fetch_qn_for_qids(qids, appns):
        need = {r[0] for r in cache.execute(
            f'SELECT 申请号 FROM q_neighbors WHERE qid IN ({",".join("?"*len(qids))})', qids)}
        need |= {a for a in appns if a}
        return _fetch_patent_texts(db_cur, need, TRUNC_N)

    def qn_of(qid, texts):
        out = [''] * N_NB
        for n_idx, a in cache.execute('SELECT n_idx, 申请号 FROM q_neighbors WHERE qid=?', (qid,)):
            out[n_idx] = texts.get(a, '')
        return out

    os.makedirs(OUT_NC, exist_ok=True)
    os.makedirs(OUT_PRE, exist_ok=True)
    rng = random.Random(42)

    # ---- 检索 train/val
    def write_retrieval(split, path, pure_bm25, n_neg, sample=None):
        qrows = cache.execute('SELECT qid, 申请号, q_text FROM queries WHERE split=? ORDER BY qid',
                              (split,)).fetchall()
        if sample and len(qrows) > sample:
            qrows = rng.sample(qrows, sample)
            qrows.sort(key=lambda x: x[0])
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(0, len(qrows), 20000):
                chunk = qrows[i:i + 20000]
                texts = fetch_qn_for_qids([x[0] for x in chunk], [x[1] for x in chunk])
                for qid, appn, qt in chunk:
                    pos = [r[0] for r in cache.execute('SELECT doc_id FROM q_pos WHERE qid=?', (qid,)).fetchall()]
                    bm = [r[0] for r in cache.execute(
                        'SELECT doc_id FROM bm25_neg WHERE qid=? ORDER BY rank LIMIT ?', (qid, n_neg)).fetchall()]
                    if pure_bm25:
                        negs = bm
                    else:
                        n_bm = min(N_BM25_HARD, len(bm))
                        bm_part = bm[:n_bm]
                        excluded = set(pos) | set(bm)
                        all_docs = list(doc_ktext.keys())
                        rand_part = []
                        while len(rand_part) < min(N_RAND, len(all_docs)):
                            cand = rng.choice(all_docs)
                            if cand not in excluded:
                                rand_part.append(cand)
                                excluded.add(cand)
                        negs = bm_part + rand_part
                    rec = {
                        'q_text': (qt or '')[:TRUNC_Q],
                        'q_n_text': qn_of(qid, texts),
                        # k 侧一律不给邻居: 正例填 kn 而负例留空会造成确定性标签泄漏
                        # (邻居 mask 是模型显式输入, 读 mask 即可分出正负例),
                        # 且 documents.txt 建索引时企业本就没有邻居, 训练/推理需一致。
                        # 企业侧图谱知识由预训练承载(见下方 write_pretrain)。勿改回 kn.get(d, ...)
                        'positives': [{'k_text': doc_ktext_full[d], 'k_n_text': [''] * N_NB} for d in pos],
                        'negatives': [{'k_text': doc_ktext[d], 'k_n_text': [''] * N_NB} for d in negs],
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'写入 {os.path.relpath(path, PROJ_DIR)}: {len(qrows)} 条')

    write_retrieval('train', os.path.join(OUT_NC, 'train.text.jsonl'), False, N_NEGS_RERANK)
    write_retrieval('val', os.path.join(OUT_NC, 'val.text.jsonl'), False, N_NEGS_RERANK)
    write_retrieval('train', os.path.join(OUT_NC, 'train.rerank.32.text.jsonl'), True, N_NEGS_RERANK,
                    sample=RERANK_TRAIN_SAMPLE)
    write_retrieval('val', os.path.join(OUT_NC, 'val.rerank.32.text.jsonl'), True, N_NEGS_RERANK)

    # ---- 测试集(全部 2024)
    qrows = cache.execute('SELECT qid, 申请号, q_text FROM queries WHERE split=? ORDER BY qid',
                          ('test',)).fetchall()
    with open(os.path.join(OUT_NC, 'test.rerank.10000.text.jsonl'), 'w', encoding='utf-8') as fr, \
         open(os.path.join(OUT_NC, 'test.node.text.jsonl'), 'w', encoding='utf-8') as fn, \
         open(os.path.join(OUT_NC, 'test.node.text.tsv'), 'w', encoding='utf-8') as ft, \
         open(os.path.join(OUT_NC, 'test.truth.trec'), 'w', encoding='utf-8') as ftr:
        for i in range(0, len(qrows), 20000):
            chunk = qrows[i:i + 20000]
            texts = fetch_qn_for_qids([x[0] for x in chunk], [x[1] for x in chunk])
            for qid, appn, qt in chunk:
                pos = [r[0] for r in cache.execute('SELECT doc_id FROM q_pos WHERE qid=?', (qid,)).fetchall()]
                bm = [r[0] for r in cache.execute(
                    'SELECT doc_id FROM bm25_neg WHERE qid=? ORDER BY rank LIMIT ?', (qid, N_NEGS_RERANK)).fetchall()]
                rec = {
                    'q_text': (qt or '')[:TRUNC_Q],
                    'q_n_text': qn_of(qid, texts),
                    # 同上: k 侧一律不给邻居, 否则重排测试集也带泄漏, 终指标会虚高
                    'positives': [{'k_text': doc_ktext_full[d], 'k_n_text': [''] * N_NB} for d in pos],
                    'negatives': [{'k_text': doc_ktext[d], 'k_n_text': [''] * N_NB} for d in bm],
                }
                fr.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fn.write(json.dumps({'id': str(qid), 'text': (qt or '')[:TRUNC_Q],
                                     'n_text': qn_of(qid, texts)}, ensure_ascii=False) + '\n')
                ft.write(f'{qid}\t{(qt or "")[:TRUNC_Q]}\n')
                for d in pos:
                    ftr.write(f'{qid} 0 {d} 1\n')
    print(f'测试集: {len(qrows)} 条')

    # ---- node_text.tsv(全部查询)
    with open(os.path.join(OUT_NC, 'node_text.tsv'), 'w', encoding='utf-8') as f:
        for qid, qt in cache.execute('SELECT qid, q_text FROM queries ORDER BY qid'):
            f.write(f'{qid}\t{(qt or "")[:TRUNC_Q]}\n')
    print('node_text.tsv 完成', flush=True)

    # ---- 预训练(<=2023 池内链接对, 流式)
    codes = dict(cache.execute('SELECT code, doc_id FROM doc_map'))
    cur = db.cursor()
    cur.execute('CREATE TEMP TABLE pool(code TEXT PRIMARY KEY)')
    cur.executemany('INSERT INTO pool VALUES(?)', [(c,) for c in codes.keys()])
    # 每专利 -> doc_ids
    cur.execute('''SELECT l.申请号, l.信用代码 FROM links_g l
        JOIN pool ON pool.code = l.信用代码
        JOIN patents_g p ON p.申请号 = l.申请号
        WHERE p.申请年份 <= 2023 ORDER BY l.申请号''')
    patent_docs = defaultdict(list)
    for appn, code in cur:
        patent_docs[appn].append(codes[code])
    print(f'预训练链接专利数: {len(patent_docs)}')

    items = list(patent_docs.items())
    rng.shuffle(items)
    val_items = items[:max(1, len(items) // 100)]
    train_items = items[max(1, len(items) // 100):]

    def write_pretrain(items, path):
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(0, len(items), 5000):
                chunk = items[i:i + 5000]
                q_texts = _fetch_patent_texts(db_cur, [a for a, _ in chunk], TRUNC_Q)
                need = set()
                qn_rows = defaultdict(list)
                for appn, _ in chunk:
                    for n_idx, nb in cache.execute('SELECT n_idx, neighbor FROM pretrain_qn WHERE 申请号=?', (appn,)):
                        qn_rows[appn].append((n_idx, nb))
                        need.add(nb)
                nb_texts = _fetch_patent_texts(db_cur, need, TRUNC_N)
                for appn, docset in chunk:
                    qn = [''] * N_NB
                    for n_idx, nb in qn_rows.get(appn, []):
                        qn[n_idx] = nb_texts.get(nb, '')
                    for d in docset:
                        rec = {'q_text': q_texts.get(appn, ''), 'k_text': doc_ktext_full[d],
                               'q_n_text': qn, 'k_n_text': kn.get(d, [''] * N_NB)}
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'预训练 {os.path.relpath(path, PROJ_DIR)}: {len(items)} 专利')

    write_pretrain(train_items, os.path.join(OUT_PRE, 'train.text.jsonl'))
    write_pretrain(val_items, os.path.join(OUT_PRE, 'val.text.jsonl'))
    db.close()
    cache.close()
    print('assemble 全部完成', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['documents', 'queries', 'neighbors', 'bm25', 'assemble'])
    args = parser.parse_args()
    globals()[f'stage_{args.stage}']()


if __name__ == '__main__':
    main()
