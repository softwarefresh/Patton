# 重排消融:去预训练 —— 底座直接用 chinese-roberta（跳过专利预训练+检索训练），其余与主实验完全一致
# 对比方式: 与主模型(有预训练)在同一个 val.rerank.small 上各测一次（见 nc_rerank_ablation_test_patent.sh）
PROJ_DIR=/workspace/Patton
cd $PROJ_DIR/src

PROCESSED_DIR=$PROJ_DIR/data/patent/nc
LOG_DIR=$PROJ_DIR/logs/patent/nc_rerank_nopretrain
CHECKPOINT_DIR=$PROJ_DIR/ckpt/patent/nc_rerank_nopretrain

LR="1e-5"
MODEL_TYPE=graphformer

# 与主实验唯一区别: 底座从 nc_retrieval 换成 chinese-roberta（图聚合层随机初始化, BERT 主体加载底座权重）
MODEL_DIR=$PROJ_DIR/ckpt/chinese-roberta-wwm-ext

echo "start rerank ablation (no pretrain) training..."

CUDA_VISIBLE_DEVICES=0 python -u -m OpenLP.driver.train_neg  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name $PROJ_DIR/ckpt/chinese-roberta-wwm-ext \
    --model_type $MODEL_TYPE \
    --do_train  \
    --hn_num 4 \
    --save_steps 500  \
    --eval_steps 1000  \
    --logging_steps 100 \
    --train_path $PROCESSED_DIR/train.rerank.32.jsonl  \
    --eval_path $PROCESSED_DIR/val.rerank.small.jsonl  \
    --fp16  \
    --per_device_train_batch_size 1  \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --dataloader_num_workers 4 \
    --learning_rate $LR  \
    --max_len 256  \
    --num_train_epochs 1  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --seed 42
