import os

# for wordLSTM target
# command = 'python attack_classification.py --dataset_path data/yelp ' \
#           '--target_model wordLSTM --batch_size 128 ' \
#           '--target_model_path /scratch/jindi/adversary/BERT/results/yelp ' \
#           '--word_embeddings_path /data/medg/misc/jindi/nlp/embeddings/glove.6B/glove.6B.200d.txt ' \
#           '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /scratch/jindi/tf_cache'

# for BERT target
import os
# 使用第一张与第三张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# imdb
command_bert_imdb = 'python attack_classification.py --dataset_path data/imdb ' \
          '--target_model bert ' \
          '--target_model_path /virtual-machine/zjyuan2.0/TextWatermark-master/BERT/Pretrained_bert/imdb ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path ./counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
          '--USE_cache_path ./tf_cache'

# yelp
command_bert_yelp = 'python attack_classification.py --dataset_path data/yelp ' \
          '--target_model bert ' \
          '--target_model_path /virtual-machine/zjyuan2.0/TextWatermark-master/BERT/Pretrained_bert/yelp ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path ./counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
          '--USE_cache_path ./tf_cache'

# mr
command_bert_mr = 'python attack_classification.py --dataset_path data/mr ' \
          '--target_model bert ' \
          '--target_model_path /virtual-machine/zjyuan2.0/TextWatermark-master/BERT/Pretrained_bert/mr ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path ./counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
          '--USE_cache_path ./tf_cache'

# ag
command_bert_ag = 'python attack_classification.py --dataset_path data/ag ' \
          '--target_model bert ' \
          '--target_model_path /virtual-machine/zjyuan2.0/TextWatermark-master/BERT/Pretrained_bert/ag ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path ./counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
          '--USE_cache_path ./tf_cache'

# fake
command_bert_fake = 'python attack_classification.py --dataset_path data/fake ' \
          '--target_model bert ' \
          '--target_model_path /virtual-machine/zjyuan2.0/TextWatermark-master/BERT/Pretrained_bert/fake ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path ./counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
          '--USE_cache_path ./tf_cache'

# os.system(command_bert_imdb)
# os.system(command_bert_yelp)
# os.system(command_bert_mr)
# os.system(command_bert_ag)
os.system(command_bert_fake)

# imdb
command_lstm_imdb = 'python attack_classification.py --dataset_path data/imdb ' \
          '--target_model wordLSTM --batch_size 128 ' \
          '--target_model_path /virtual-machine/zjyuan2.0/TextWatermark-master/BERT/Pretrained_lstm/imdb ' \
          '--word_embeddings_path /virtual-machine/zjyuan2.0/TextWatermark-master/data/glove.6B.200d.txt ' \
          '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
          '--USE_cache_path /scratch/jindi/tf_cache'

# yelp
command_lstm_yelp = 'python attack_classification.py --dataset_path data/yelp ' \
          '--target_model wordLSTM --batch_size 128 ' \
          '--target_model_path /virtual-machine/zjyuan2.0/TextWatermark-master/BERT/Pretrained_lstm/yelp ' \
          '--word_embeddings_path /virtual-machine/zjyuan2.0/TextWatermark-master/data/glove.6B.200d.txt ' \
          '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
          '--USE_cache_path /scratch/jindi/tf_cache'

# os.system(command_lstm_imdb)
# os.system(command_lstm_fake)
# os.system(command_lstm_yelp)