# 检索阶段训练 v2 (专利 -> 企业): 恢复 in-batch 负例
# 与 ②' 的区别: max_len 256→128 换 batch 1→4; hn_num 4→3（每个负例占 6 条子图序列,
# 少 1 个负例省 1/6 显存——batch 4@128+hn4 实测 OOM 10.9G, hn3 ≈9G 可装）
# 效果: 每个 micro-batch 有 4 个 in-batch 负例 + 3 BM25 硬负 = 7 路对比;
# ②' 是 1 正 + 4 负 = 5 路, 从没学过全库判别（recall@100 只有 0.43 的根因之一）
# 若仍 OOM, 回退: batch 2 + hn_num 4（6 路, 显存 ~5.5G）
# 输出独立目录 nc_retrieval_v2, 不覆盖 ②'（④ 重排的底座）
PROJ_DIR=/workspace/Patton
cd $PROJ_DIR/src

PROCESSED_DIR=$PROJ_DIR/data/patent/nc
LOG_DIR=$PROJ_DIR/logs/patent/nc_retrieval_v2
CHECKPOINT_DIR=$PROJ_DIR/ckpt/patent/nc_retrieval_v2

LR="1e-5"
MODEL_TYPE=graphformer

MODEL_DIR=$PROJ_DIR/ckpt/patent/pretrain/graphformer/$LR

echo "start retrieval v2 training (in-batch negatives)..."

CUDA_VISIBLE_DEVICES=0 python -u -m OpenLP.driver.train_neg  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name $PROJ_DIR/ckpt/chinese-roberta-wwm-ext \
    --model_type $MODEL_TYPE \
    --do_train  \
    --hn_num 3 \
    --save_steps 500  \
    --eval_steps 1000  \
    --logging_steps 100 \
    --train_path $PROCESSED_DIR/train.half.jsonl  \
    --eval_path $PROCESSED_DIR/val.small.jsonl  \
    --fp16  \
    --per_device_train_batch_size 4  \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --dataloader_num_workers 4 \
    --learning_rate $LR  \
    --max_len 128  \
    --num_train_epochs 1  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --seed 42
