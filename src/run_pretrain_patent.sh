# 专利预训练: 中文底座 -> 专利语料继续预训练 (MLM + 对比)
# 单卡 3080 Ti (12GB): fp16 + max_len 256 + batch 4×累积32
# 不用 grad_cache：GradCache 路径下 MLM 损失会被静默丢弃，预训练必须完整走对比+MLM
PROJ_DIR=/workspace/Patton
cd $PROJ_DIR/src

PROCESSED_DIR=$PROJ_DIR/data/patent/pretrain
LOG_DIR=$PROJ_DIR/logs/patent/pretrain
CHECKPOINT_DIR=$PROJ_DIR/ckpt/patent/pretrain

LR="1e-5"
MODEL_TYPE=graphformer

export CUDA_VISIBLE_DEVICES=0

echo "start pretraining..."

python -m OpenLP.driver.patton_pretrain  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $PROJ_DIR/ckpt/chinese-roberta-wwm-ext  \
    --model_type $MODEL_TYPE \
    --do_train  \
    --save_steps 20000  \
    --eval_steps 10000  \
    --logging_steps 1000 \
    --train_path $PROCESSED_DIR/train.jsonl  \
    --eval_path $PROCESSED_DIR/val.jsonl  \
    --fp16  \
    --per_device_train_batch_size 4  \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --learning_rate $LR  \
    --max_len 256  \
    --num_train_epochs 2  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --mlm_loss True \
    --dataloader_num_workers 4
