#!/bin/bash

# Run this command and analyse the tensorboard logs.
python3 train.py probing \
	data/imdb_sentiment_train_5k.jsonl \
	data/imdb_sentiment_dev.jsonl \
	--base-model-dir serialization_dirs/main_dan_5k_with_emb \
	--num-epochs 5 \
	--layer-num 3
