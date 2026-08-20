domain=amazon
dataset=sports

# retrieval for all data samples (prepare hard negatives)
python bm25.py --domain $domain --dataset $dataset --k 20 --mode all

# # retrieval for test data samples
# python bm25.py --domain $domain --dataset $dataset --k 200 --mode test

# bm25/trec_eval/trec_eval -c -m recip_rank.1 -m P.1 ../data/${domain}/${dataset}/nc/test.truth.trec ../data/${domain}/${dataset}/nc/bm25_test_trec
# bm25/trec_eval/trec_eval -c -m P.5 ../data/${domain}/${dataset}/nc/test.truth.trec ../data/${domain}/${dataset}/nc/bm25_test_trec

# bm25/trec_eval/trec_eval -c -m recall.20 ../data/${domain}/${dataset}/nc/test.truth.trec ../data/${domain}/${dataset}/nc/bm25_test_trec
# bm25/trec_eval/trec_eval -c -m recall.50 ../data/${domain}/${dataset}/nc/test.truth.trec ../data/${domain}/${dataset}/nc/bm25_test_trec
# bm25/trec_eval/trec_eval -c -m recall.100 ../data/${domain}/${dataset}/nc/test.truth.trec ../data/${domain}/${dataset}/nc/bm25_test_trec
# bm25/trec_eval/trec_eval -c -m recall.200 ../data/${domain}/${dataset}/nc/test.truth.trec ../data/${domain}/${dataset}/nc/bm25_test_trec
