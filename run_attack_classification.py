import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# # for wordLSTM target
# command = 'python attack_classification.py --dataset_path data/yelp ' \
#           '--target_model wordLSTM --batch_size 128 ' \
#           '--target_model_path ./LSTM/yelp ' \
#           '--word_embeddings_path ./glove.6B.200d.txt ' \
#           '--counter_fitting_embeddings_path ./counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path ./tf_cache'

# CNN
# command = 'python attack_classification.py --dataset_path data/imdb ' \
#           '--target_model wordCNN --batch_size 128 ' \
#           '--target_model_path ./CNN/imdb ' \
#           '--word_embeddings_path ./glove.6B.200d.txt ' \
#           '--counter_fitting_embeddings_path ./counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path ./tf_cache'

# for BERT target
command = 'python attack_classification.py --dataset_path data/imdb ' \
          '--data_size 1000 ' \
          '--target_model bert ' \
          '--target_model_path /virtual-machine/zjyuan2.0/TextWatermark-master3.0/BERT/imdb ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path ./counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
          '--USE_cache_path ./tf_cache'


os.system(command)
