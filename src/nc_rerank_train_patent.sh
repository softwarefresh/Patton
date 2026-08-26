# 重排阶段训练
PROJ_DIR=/workspace/Patton
cd $PROJ_DIR/src

PROCESSED_DIR=$PROJ_DIR/data/patent/nc
LOG_DIR=$PROJ_DIR/logs/patent/nc_rerank
CHECKPOINT_DIR=$PROJ_DIR/ckpt/patent/nc_rerank

LR="1e-5"
MODEL_TYPE=graphformer

MODEL_DIR=$PROJ_DIR/ckpt/patent/nc_retrieval/$MODEL_TYPE/$LR
# 冒烟时可直接用底座:
# MODEL_DIR=$PROJ_DIR/ckpt/chinese-roberta-wwm-ext

echo "start rerank training..."

# 显存适配(12G): 每样本36条子图序列(查询6+正例6+负例4×6空占位)，max_len 256 下每条约175MB，12G 只能 batch 1；
# batch 1×累积128=有效128；不用 grad_cache（max_len 256 下前向缓存同样爆）
# -u: stdout 无缓冲，nohup 重定向下 loss 日志实时落盘（默认块缓冲会攒到进程结束才写出）
CUDA_VISIBLE_DEVICES=0 python -u -m OpenLP.driver.train_neg  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name $PROJ_DIR/ckpt/chinese-roberta-wwm-ext \
    --model_type $MODEL_TYPE \
    --do_train  \
    --hn_num 4 \
    --save_steps 1000  \
    --eval_steps 1000  \
    --logging_steps 100 \
    --train_path $PROCESSED_DIR/train.rerank.32.jsonl  \
    --eval_path $PROCESSED_DIR/val.rerank.32.jsonl  \
    --fp16  \
    --per_device_train_batch_size 1  \
    --per_device_eval_batch_size 16 \
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
