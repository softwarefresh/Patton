# 检索阶段训练 (专利 -> 企业)
# 底座: 预训练后的 patton checkpoint; 若预训练未完成先用 chinese-roberta-wwm-ext 跑通
PROJ_DIR=/workspace/Patton
cd $PROJ_DIR/src

PROCESSED_DIR=$PROJ_DIR/data/patent/nc
LOG_DIR=$PROJ_DIR/logs/patent/nc_retrieval
CHECKPOINT_DIR=$PROJ_DIR/ckpt/patent/nc_retrieval

LR="1e-5"
MODEL_TYPE=graphformer

MODEL_DIR=$PROJ_DIR/ckpt/patent/pretrain/graphformer/$LR
# 冒烟/无预训练时用:
# MODEL_DIR=$PROJ_DIR/ckpt/chinese-roberta-wwm-ext

echo "start retrieval training..."

CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.train_neg  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name $PROJ_DIR/ckpt/chinese-roberta-wwm-ext \
    --model_type $MODEL_TYPE \
    --do_train  \
    --hn_num 4 \
    --save_steps 5000  \
    --eval_steps 5000  \
    --logging_steps 500 \
    --train_path $PROCESSED_DIR/train.jsonl  \
    --eval_path $PROCESSED_DIR/val.jsonl  \
    --fp16  \
    --grad_cache  \
    --per_device_train_batch_size 8  \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --learning_rate $LR  \
    --max_len 256  \
    --num_train_epochs 2  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --seed 42
