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
parser.add_argument('--max_length', type=int, default=256)
args = parser.parse_args()

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

save_dir = os.path.split(args.output)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

files = os.listdir(args.input_dir)

def process(item):

    group = {}

    query = tokenizer.encode(item['q_text'], add_special_tokens=False, max_length=args.max_length, truncation=True)
    key = tokenizer.encode(item['k_text'], add_special_tokens=False, max_length=args.max_length, truncation=True)
    q_n_text = tokenizer(
        item['q_n_text'], add_special_tokens=False, max_length=args.max_length, truncation=True, padding=False)['input_ids']
    k_n_text = tokenizer(
        item['k_n_text'], add_special_tokens=False, max_length=args.max_length, truncation=True, padding=False)['input_ids']

    group['q_text'] = query
    group['k_text'] = key
    group['q_n_text'] = q_n_text
    group['k_n_text'] = k_n_text

    return json.dumps(group)


# multiprocessing mode
# 流式逐行处理 + 限制 worker 数，避免把大文件整体载入内存被 OOM kill
def iter_lines(path):
    with open(path) as fin:
        for line in fin:
            yield json.loads(line)


for split in ['train', 'val', 'test']:
    in_file = os.path.join(args.input_dir, f'{split}.text.jsonl')
    if not os.path.exists(in_file):
        continue
    out_file = os.path.join(args.output, f'{split}.jsonl')
    pool = Pool(processes=args.mp_workers)
    with open(out_file, 'w') as f:
        pbar = tqdm(iter_lines(in_file))
        for x in pool.imap(process, pbar, chunksize=args.mp_chunk_size):
            if x != 0:
                f.write(x + '\n')
    pool.close()
    pool.join()
