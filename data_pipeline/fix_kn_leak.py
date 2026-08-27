"""修复检索/重排数据里的标签泄漏：正例有企业侧邻居、负例全空。

问题：build 出来的 *.text.jsonl 中，positives 的 k_n_text 是真实内容，
negatives 的 k_n_text 五条全是空串。而 mask 是模型的显式输入通道
（train_dataset.py:128 `1 if k_n != [] else 0`），模型直接读 mask 就能
100% 分出正负例，训练损失被压到 0，学不到任何语义。

同时 FAISS 建索引用的 documents.txt 只有 id/text 两列，
inference_dataset.py:53 的 `if 'n_text' in example` 走不进去，
25,598 家企业全部零邻居入库——训练和推理口径也对不上。

修法（对齐上游 Patton）：正例的 k_n_text 也置空，k 侧全员无邻居。
企业侧图谱知识由预训练承载（那边 q/k 两侧都有邻居，不受影响）；
检索阶段的图信息走查询侧 q_n_text，与推理端 test.node.text.jsonl 一致。

注意：空邻居仍占一个子图序列槽（train_dataset.py:127 用 [0] 占位），
所以置空必须**保留原有槽位数**，不能把列表清成 []，否则张量形状对不上。

用法（默认 dry-run 只统计，不动文件）：
    python data_pipeline/fix_kn_leak.py --dir data/patent/nc
    python data_pipeline/fix_kn_leak.py --dir data/patent/nc --apply
"""

import argparse
import json
import os
import sys
from collections import Counter


def is_candidate_file(path):
    """只处理 {q_text, q_n_text, positives, negatives} 这种检索/重排格式。

    query 侧的 test.node.text.jsonl（id/text/n_text）会被排除。
    """
    try:
        with open(path, encoding='utf-8') as f:
            first = f.readline()
        if not first.strip():
            return False
        rec = json.loads(first)
    except (OSError, ValueError):
        return False
    return isinstance(rec, dict) and 'positives' in rec and 'negatives' in rec


def scan_group(group, stats, side):
    """统计一组候选的邻居填充情况。"""
    for cand in group:
        kn = cand.get('k_n_text')
        if kn is None:
            stats[f'{side}_missing_field'] += 1
            continue
        stats[f'{side}_total'] += 1
        stats[f'{side}_slots_{len(kn)}'] += 1
        if any(t != '' for t in kn):
            stats[f'{side}_nonempty'] += 1


def process_file(path, apply_changes, force):
    """单次流式扫描：统计 + （可选）改写到临时文件后原子替换。"""
    stats = Counter()
    tmp_path = path + '.tmp'
    fout = open(tmp_path, 'w', encoding='utf-8', newline='\n') if apply_changes else None

    try:
        with open(path, encoding='utf-8') as fin:
            for lineno, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                stats['records'] += 1

                scan_group(rec.get('positives', []), stats, 'pos')
                scan_group(rec.get('negatives', []), stats, 'neg')

                if apply_changes:
                    for cand in rec.get('positives', []):
                        kn = cand.get('k_n_text')
                        if kn:
                            # 保留槽位数，只把内容清空
                            cand['k_n_text'] = [''] * len(kn)
                    fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

                if stats['records'] % 20000 == 0:
                    print(f'    ... {stats["records"]:,} 条', flush=True)
    except Exception:
        if fout:
            fout.close()
            os.remove(tmp_path)
        raise

    if fout:
        fout.close()

    # 前提校验：负例本来就该全空。若不是，说明数据形态和分析不符，不要盲改。
    violated = stats['neg_nonempty'] > 0
    if apply_changes:
        if violated and not force:
            os.remove(tmp_path)
            print(f'    !! 中止：有 {stats["neg_nonempty"]:,} 个负例带非空邻居，'
                  f'与「负例全空」的前提不符。原文件未改动。'
                  f'确认无误可加 --force 强制执行。', flush=True)
            stats['aborted'] = 1
        else:
            os.replace(tmp_path, path)
            stats['rewritten'] = 1

    return stats


def report(name, stats):
    pos_t, pos_n = stats['pos_total'], stats['pos_nonempty']
    neg_t, neg_n = stats['neg_total'], stats['neg_nonempty']
    print(f'  记录数        : {stats["records"]:,}')
    print(f'  正例          : {pos_t:,} 个，其中带非空邻居 {pos_n:,} '
          f'({pos_n / pos_t * 100:.1f}%)' if pos_t else '  正例          : 0')
    print(f'  负例          : {neg_t:,} 个，其中带非空邻居 {neg_n:,} '
          f'({neg_n / neg_t * 100:.1f}%)' if neg_t else '  负例          : 0')

    slots = {k.rsplit('_', 1)[-1]: v for k, v in stats.items() if k.startswith('pos_slots_')}
    neg_slots = {k.rsplit('_', 1)[-1]: v for k, v in stats.items() if k.startswith('neg_slots_')}
    print(f'  邻居槽位分布  : 正例 {dict(sorted(slots.items()))} / 负例 {dict(sorted(neg_slots.items()))}')

    if slots.keys() != neg_slots.keys():
        print('  !! 警告：正负例槽位数不一致，子图张量形状可能对不上')
    if stats['pos_missing_field'] or stats['neg_missing_field']:
        print(f'  !! 警告：缺 k_n_text 字段 正例{stats["pos_missing_field"]} '
              f'负例{stats["neg_missing_field"]}')

    if pos_t and neg_t:
        if pos_n > 0 and neg_n == 0:
            print('  判定          : 存在确定性泄漏（正例有邻居 / 负例全空）')
        elif pos_n == 0 and neg_n == 0:
            print('  判定          : 干净（两侧都无邻居，与推理端一致）')
        else:
            print('  判定          : 两侧都有邻居，需人工确认')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='data/patent/nc',
                    help='存放 *.text.jsonl 的目录（默认 data/patent/nc）')
    ap.add_argument('--files', nargs='*', default=None,
                    help='只处理指定文件名（默认自动发现目录下所有候选格式文件）')
    ap.add_argument('--apply', action='store_true',
                    help='真正改写文件；不加则只统计（dry-run）')
    ap.add_argument('--force', action='store_true',
                    help='即使负例带非空邻居也强制改写')
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f'目录不存在：{args.dir}')

    if args.files:
        names = args.files
    else:
        names = sorted(n for n in os.listdir(args.dir) if n.endswith('.text.jsonl'))

    targets = []
    for n in names:
        p = os.path.join(args.dir, n)
        if is_candidate_file(p):
            targets.append(p)
        else:
            print(f'跳过 {n}（不是 positives/negatives 格式）')

    if not targets:
        sys.exit('没有找到可处理的文件')

    mode = '改写模式 (--apply)' if args.apply else 'dry-run（只统计，不改文件）'
    print(f'\n=== {mode} ===')
    print(f'目标目录：{os.path.abspath(args.dir)}')
    print(f'待处理  ：{len(targets)} 个文件\n')

    total = Counter()
    for p in targets:
        size_gb = os.path.getsize(p) / 1024 ** 3
        print(f'[{os.path.basename(p)}] {size_gb:.1f} GB', flush=True)
        stats = process_file(p, args.apply, args.force)
        report(os.path.basename(p), stats)
        if stats['rewritten']:
            print('  -> 已改写')
        total.update(stats)
        print(flush=True)

    print('=== 汇总 ===')
    print(f'处理文件 {len(targets)} 个，记录 {total["records"]:,} 条')
    print(f'正例带非空邻居 {total["pos_nonempty"]:,} / {total["pos_total"]:,}')
    print(f'负例带非空邻居 {total["neg_nonempty"]:,} / {total["neg_total"]:,}')
    if args.apply:
        print(f'已改写 {total["rewritten"]} 个文件，中止 {total["aborted"]} 个')
        print('\n下一步：重新 tokenize（跳过预训练部分，见 PROGRESS.md）')
    else:
        print('\n这是 dry-run。确认统计无误后加 --apply 执行。')


if __name__ == '__main__':
    main()
