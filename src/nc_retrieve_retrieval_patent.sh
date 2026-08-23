# 检索: 用测试查询搜索, 算 recall 指标
PROJ_DIR=/workspace/Patton
cd $PROJ_DIR/src

MODEL_TYPE=graphformer
LR=1e-5

DATA_DIR=$PROJ_DIR/data/patent
CHECKPOINT_DIR=$PROJ_DIR/ckpt/patent/nc_retrieval/$MODEL_TYPE/$LR

echo "running search..."

CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.search  \
    --output_dir $CHECKPOINT_DIR/node_label_embed  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --tokenizer_name $PROJ_DIR/ckpt/chinese-roberta-wwm-ext \
    --model_type $MODEL_TYPE \
    --per_device_eval_batch_size 16  \
    --corpus_path $DATA_DIR/nc/documents.txt  \
    --query_path $DATA_DIR/nc/test.node.text.jsonl  \
    --query_column_names 'id,text' \
    --max_len 256 \
    --save_trec True \
    --retrieve_domain patent \
    --source_domain patent \
    --save_path $DATA_DIR/nc/retrieve_trec  \
    --dataloader_num_workers 4

echo "calculating metrics..."
python scripts/eval_trec.py $DATA_DIR/nc/test.truth.trec $DATA_DIR/nc/retrieve_trec --k 50 100

rm $DATA_DIR/nc/retrieve_trec
rm $DATA_DIR/nc/patent_patent_retrieval_dict.pkl
