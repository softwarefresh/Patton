"""计算检索 recall@K（替代原版依赖的 trec_eval 二进制, 纯 Python 无依赖）。

truth.trec 格式: qid 0 docid 1
检索结果格式:   qid Q0 docid rank score run_name (按 rank 升序)
用法: python eval_trec.py truth.trec retrieve_trec --k 50 100
"""
import argparse
from collections import defaultdict


def load_truth(path):
    truth = defaultdict(set)
    for line in open(path):
        parts = line.strip().split()
        if len(parts) >= 4:
            truth[parts[0]].add(parts[2])
    return truth


def load_retrieve(path):
    res = defaultdict(list)
    for line in open(path):
        parts = line.strip().split()
        if len(parts) >= 6:
            res[parts[0]].append((int(parts[3]), parts[2]))
    for qid in res:
        res[qid].sort()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('truth')
    ap.add_argument('retrieve')
    ap.add_argument('--k', type=int, nargs='+', default=[50, 100])
    args = ap.parse_args()

    truth = load_truth(args.truth)
    res = load_retrieve(args.retrieve)

    totals = {k: 0.0 for k in args.k}
    n = 0
    for qid, t in truth.items():
        if qid not in res or not t:
            continue
        n += 1
        for k in args.k:
            hits = sum(1 for _, d in res[qid][:k] if d in t)
            totals[k] += hits / len(t)
    for k in args.k:
        print(f'recall@{k} = {totals[k] / n:.4f}  (n={n})')


if __name__ == '__main__':
    main()
