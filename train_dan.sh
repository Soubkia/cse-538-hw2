#!/bin/bash

# Run this command and analyse the tensorboard logs.
python3 train.py main \
	data/imdb_sentiment_train_5k.jsonl \
	data/imdb_sentiment_dev.jsonl \
	--seq2vec-choice dan \
	--embedding-dim 50 \
	--num-layers 4 \
	--num-epochs 5 \
	--suffix-name _dan_5k_with_emb \
	--pretrained-embedding-file data/glove.6B.50d.txt
