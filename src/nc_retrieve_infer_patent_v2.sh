# 检索 v2: 对候选企业语料建稠密向量索引 (documents.txt)
# 注意: 输出在 v2 自己的 node_label_embed 目录, 不覆盖 ②' 的向量
PROJ_DIR=/workspace/Patton
cd $PROJ_DIR/src

MODEL_TYPE=graphformer
LR=1e-5

CHECKPOINT_DIR=$PROJ_DIR/ckpt/patent/nc_retrieval_v2/$MODEL_TYPE/$LR

echo "running infer (v2)..."

CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.infer  \
    --output_dir $CHECKPOINT_DIR/node_label_embed  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --tokenizer_name $PROJ_DIR/ckpt/chinese-roberta-wwm-ext \
    --model_type $MODEL_TYPE \
    --per_device_eval_batch_size 16  \
    --corpus_path $PROJ_DIR/data/patent/nc/documents.txt  \
    --doc_column_names 'id,text' \
    --max_len 128 \
    --retrieve_domain patent \
    --dataloader_num_workers 4
