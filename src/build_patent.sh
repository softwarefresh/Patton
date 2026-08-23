# 专利数据 tokenize（在服务器上跑, CPU 即可）
# 输入: data/patent/{nc,pretrain}/*.text.jsonl
# 输出: 对应目录下的 *.jsonl (token id 格式)
PROJ_DIR=/workspace/Patton
TOKENIZER=$PROJ_DIR/ckpt/chinese-roberta-wwm-ext
NC_DIR=$PROJ_DIR/data/patent/nc
PRE_DIR=$PROJ_DIR/data/patent/pretrain

cd $PROJ_DIR/src/scripts

# 预训练
python build_train.py \
    --input_dir $PRE_DIR \
    --output $PRE_DIR \
    --tokenizer $TOKENIZER \
    --max_length 256 \
    --mp_workers 8

# 检索
python build_train_neg.py \
    --input_dir $NC_DIR \
    --output $NC_DIR \
    --tokenizer $TOKENIZER \
    --max_length 256 \
    --mp_workers 8

# 重排(训练/验证)
python build_train_neg.py \
    --input_dir $NC_DIR \
    --output $NC_DIR \
    --tokenizer $TOKENIZER \
    --max_length 256 \
    --mp_workers 8 \
    --prefix rerank.32

# 重排(测试, 10000 条查询)
python build_train_neg.py \
    --input_dir $NC_DIR \
    --output $NC_DIR \
    --tokenizer $TOKENIZER \
    --max_length 256 \
    --mp_workers 8 \
    --prefix rerank.10000
