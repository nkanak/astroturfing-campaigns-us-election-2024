#!/usr/bin/env bash

rm -r produced_data/*
rm -r tweets
rm -r missing_user_profiles
rm -r tsne_social_graph
rm -r tsne_trees

echo "##### RUN dataset_preprocess #####"
python dataset_preprocess.py
echo "##### RUN create_trees #####"
python create_trees.py
echo "##### RUN split_dataset #####"
python split_dataset.py

echo "##### RUN compute_user_labels #####"
python compute_user_labels.py --input-dir=raw_data
echo "##### RUN users_to_graph #####"
python users_to_graph.py --input-dir raw_data --embeddings-file raw_data/glove.twitter.27B.100d.txt --dataset-root produced_data/datasets/dataset

echo "##### RUN train_social_graph #####"
python train_social_graph.py --dataset-root produced_data/datasets/dataset --epochs 200

echo "##### RUN compute_user_embeddings #####"
python compute_user_embeddings.py --input-dir raw_data --dataset-root produced_data/datasets/dataset --embeddings-file raw_data/glove.twitter.27B.100d.txt
echo "##### RUN add_trees_information #####"
python add_trees_information.py --dataset-root produced_data/datasets/dataset
echo "##### RUN train_trees #####"
python train_trees.py