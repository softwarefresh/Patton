# 重排消融对比测试: 主模型(有预训练) vs 去预训练 —— 同一 val.rerank.small，参数完全一致
# 不跑全量 test(38.7h); val.rerank.small ~1万条, 每个模型约 1.2h
# 注意: 主模型此前的冒烟 0.842 用的是哪个文件已不可考, 所以这里两个模型都重测, 对比才公平
PROJ_DIR=/workspace/Patton
cd $PROJ_DIR/src

MODEL_TYPE=graphformer
LR=1e-5
STEP=500
TEST_DIR=$PROJ_DIR/data/patent/nc

for MODEL in nc_rerank nc_rerank_nopretrain; do
  echo "=== testing $MODEL ==="
  CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.test_rerank  \
      --output_dir $TEST_DIR/tmp  \
      --model_name_or_path $PROJ_DIR/ckpt/patent/$MODEL/$MODEL_TYPE/$LR/checkpoint-$STEP  \
      --tokenizer_name $PROJ_DIR/ckpt/chinese-roberta-wwm-ext \
      --model_type $MODEL_TYPE \
      --do_eval  \
      --pos_rerank_num 1 \
      --neg_rerank_num 20 \
      --train_path $TEST_DIR/val.rerank.small.jsonl  \
      --eval_path $TEST_DIR/val.rerank.small.jsonl  \
      --fp16  \
      --per_device_eval_batch_size 2 \
      --eval_accumulation_steps 50 \
      --max_len 256  \
      --evaluation_strategy steps \
      --remove_unused_columns False \
      --overwrite_output_dir True \
      --dataloader_num_workers 4
done
