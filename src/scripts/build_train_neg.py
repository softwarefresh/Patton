# This code script is for node classification retrieval with bm25 negative samples only. Keep in mind.

import json
import os
from argparse import ArgumentParser

from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from multiprocessing import Pool

parser = ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--tokenizer', type=str, required=False, default='bert-base-uncased')
parser.add_argument('--minimum-negatives', type=int, required=False, default=1)
parser.add_argument('--mp_chunk_size', type=int, required=False, default=1)
parser.add_argument('--mp_workers', type=int, required=False, default=4)
parser.add_argument('--prefix', type=str, required=False, default='')
parser.add_argument('--max_length', type=int, required=False, default=256)
args = parser.parse_args()

if args.prefix != '':
    args.prefix = '.' + args.prefix

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

save_dir = os.path.split(args.output)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

files = os.listdir(args.input_dir)

def process(item):

    group = {}

    query = tokenizer.encode(item['q_text'], add_special_tokens=False, max_length=args.max_length, truncation=True)
    q_n_text = tokenizer(
        item['q_n_text'], add_special_tokens=False, max_length=args.max_length, truncation=True, padding=False)['input_ids']

    positives = []
    for k in item['positives']:
        positives.append({'k_text': tokenizer.encode(k['k_text'], add_special_tokens=False, max_length=args.max_length, truncation=True),
                        'k_n_text': tokenizer(
                                    k['k_n_text'], add_special_tokens=False, max_length=args.max_length, truncation=True, padding=False)['input_ids']})

    negatives = []
    for k in item['negatives']:
        negatives.append({'k_text': tokenizer.encode(k['k_text'], add_special_tokens=False, max_length=args.max_length, truncation=True),
                        'k_n_text': tokenizer(
                                    k['k_n_text'], add_special_tokens=False, max_length=args.max_length, truncation=True, padding=False)['input_ids']})

    # key = tokenizer.encode(item['k_text'], add_special_tokens=False, max_length=args.max_length, truncation=True)
    # k_n_text = tokenizer(
    #     item['k_n_text'], add_special_tokens=False, max_length=args.max_length, truncation=True, padding=False)['input_ids']

    group['q_text'] = query
    group['q_n_text'] = q_n_text
    group['positives'] = positives
    group['negatives'] = negatives

    return json.dumps(group)


# multiprocessing mode
# 流式逐行处理 + 限制 worker 数，避免把大文件整体载入内存被 OOM kill
def iter_lines(path):
    with open(path) as fin:
        for line in fin:
            yield json.loads(line)


for split in ['train', 'val', 'test']:
    in_file = os.path.join(args.input_dir, f'{split}{args.prefix}.text.jsonl')
    if not os.path.exists(in_file):
        continue
    out_file = os.path.join(args.output, f'{split}{args.prefix}.jsonl')
    pool = Pool(processes=args.mp_workers)
    with open(out_file, 'w') as f:
        pbar = tqdm(iter_lines(in_file))
        for x in pool.imap(process, pbar, chunksize=args.mp_chunk_size):
            if x != 0:
                f.write(x + '\n')
    pool.close()
    pool.join()
