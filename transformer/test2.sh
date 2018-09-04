

# Export variables
PARAM_SET=big
RAW_DATA_DIR=./data_open_domain_final
DATA_DIR=./data
MODEL_DIR=./model2/model_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.30k
TEST_FILE=$RAW_DATA_DIR/infer.in
TEST_FILE_OUT=./infer2.res

# Translate some text using the trained model
CUDA_VISIBLE_DEVICES=1 python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --file=$TEST_FILE --file_out=$TEST_FILE_OUT

