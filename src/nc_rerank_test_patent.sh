# 重排阶段测试 (把 $STEP 改成最佳 checkpoint 步数)
PROJ_DIR=/workspace/Patton
cd $PROJ_DIR/src

MODEL_TYPE=graphformer
LR=1e-5
STEP=500

CHECKPOINT_DIR=$PROJ_DIR/ckpt/patent/nc_rerank/$MODEL_TYPE/$LR/checkpoint-$STEP

TEST_DIR=$PROJ_DIR/data/patent/nc

echo "running rerank test..."

CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.test_rerank  \
    --output_dir $TEST_DIR/tmp  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --tokenizer_name $PROJ_DIR/ckpt/chinese-roberta-wwm-ext \
    --model_type $MODEL_TYPE \
    --do_eval  \
    --pos_rerank_num 1 \
    --neg_rerank_num 20 \
    --train_path $TEST_DIR/test.rerank.10000.jsonl  \
    --eval_path $TEST_DIR/test.rerank.10000.jsonl  \
    --fp16  \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 50 \
    --max_len 256  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 4
