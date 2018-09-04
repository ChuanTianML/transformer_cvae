# Ensure that PYTHONPATH is correctly defined as described in
# https://github.com/tensorflow/models/tree/master/official#requirements
# export PYTHONPATH="$PYTHONPATH:/path/to/models"

# Export variables
PARAM_SET=big
#PARAM_SET=base
RAW_DATA_DIR=./data_open_domain_final
DATA_DIR=./data
MODEL_DIR=./model/model_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.30k

# Download training/evaluation datasets
#python data_trans_to_tfrcd.py --data_dir=$DATA_DIR --raw_dir=$RAW_DATA_DIR

# Train the model for 10 epochs, and evaluate after every epoch.
rm -r $MODEL_DIR
CUDA_VISIBLE_DEVICES=3 python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET

:<<!
# Run during training in a separate process to get continuous updates,
# or after training is complete.
tensorboard --logdir=$MODEL_DIR

# Translate some text using the trained model
python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --text="hello world"

# Compute model's BLEU score using the newstest2014 dataset.
python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --file=test_data/newstest2014.en --file_out=translation.en
python compute_bleu.py --translation=translation.en --reference=test_data/newstest2014.de
!
