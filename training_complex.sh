#adapted from the repository for the University of Edinburgh School of Informatics Machine Learning Practical course:
#https://github.com/CSTR-Edinburgh/mlpractical/tree/mlp2018-9/mlp_cluster_tutorial

#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=3-00:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

export XDG_RUNTIME_DIR=/home/${STUDENT_ID}/Dissertation

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/


# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate dissertation

#run the file with arguments
python experiment_complex.py --epochs 15 --batch_size 16 --train_size 242610 --val_size 80870 --embedding_dim 300 --data_path "/home/s1834310/Dissertation/Data/QuoraDuplicateQuestions/" --output_filename "/home/s1834310/Dissertation/Models/quora_word2vec_lstm_siam" --use_gpu 'True' --gpu_id '0' --train_dataset_name "quora_train" --val_dataset_name "quora_val" --seed 100 --dataset_class 'RawTextDataset' --network 'LSTMSiameseNet' --embedding_type 'word2vec' --max_tokens 31 --text_column1 'question1' --text_column2 'question2' --encoding 'utf-8' --hidden_size 64 --num_layers 4 --out_features 128 --dropout 0.2 --bidirectional True --weight_decay_coefficient 0
