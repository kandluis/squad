MODEL="bert-base-uncased"

## The following are unused
# --char_emb_file
# --word_emb_file
## The folowing are left to their default
# --ans_limit 30
# --char_limit 16
# --include_test_examples True
# --max_seq_length 384
# --doc_stride 128
# --max_query_length 64
python setup_bert.py \
  --train_record_file ./data/train-$MODEL.npz \
  --dev_record_file ./data/dev-$MODEL.npz \
  --test_record_file ./data/test-$MODEL.npz \
  --train_eval_file ./data/train_eval-$MODEL.json \
  --dev_eval_file ./data/dev_eval-$MODEL.json\
  --test_eval_file ./data/test_eval-$MODEL.json \
  --bert_model $MODEL \
  --train_file ./data/train-v2.0.json \
  --dev_file ./data/dev-v2.0.json \
  --test_file ./data/test-v2.0.json

