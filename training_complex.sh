#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

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


#export STUDENT_ID=$(whoami) # Create the directory on the scratch disk 
#mkdir -p /disk/scratch/${STUDENT_ID} 
#export TMPDIR=/disk/scratch/${STUDENT_ID}/ 
#export TMP=/disk/scratch/${STUDENT_ID} # Create a subdirectory for your dataset 
#mkdir -p ${TMP}/data/ # Copy over the dataset (I'd recommend copying it in zip format, and then unzipping, otherwise the copying will take forever) 
rsync -ua --progress /home/${STUDENT_ID}/Dissertation/fasttext.vectors.npy ${TMP}
rsync -ua --progress /home/${STUDENT_ID}/Dissertation/word2vec.vectors.npy ${TMP}

# Activate the relevant virtual environment:


source /home/${STUDENT_ID}/miniconda3/bin/activate dissertation

#this is what Marko has
# data_path = "/afs/inf.ed.ac.uk/user/s18/s1834310/Documents/Dissertation/Data/QuoraDuplicateQuestions"
# target_path = "/afs/inf.ed.ac.uk/user/s18/s1834310/Documents/Dissertation/Models"#is this where things will get output??
# mkdir -p ${target_dir}
# rsync -ua --progress data_dir target_dir
#but I may just not worry about this for now

#python3 or just python?
python experiment_complex.py --epochs 15 --batch_size 16 --train_size 242610 --val_size 80870 --embedding_dim 300 --data_path "/home/s1834310/Dissertation/Data/QuoraDuplicateQuestions/" --output_filename "/home/s1834310/Dissertation/Models/quora_word2vec_lstm_siam" --use_gpu 'True' --gpu_id '0' --train_dataset_name "quora_train" --val_dataset_name "quora_val" --seed 100 --dataset_class 'RawTextDataset' --network 'LSTMSiameseNet' --embedding_type 'word2vec' --max_tokens 31 --text_column1 'question1' --text_column2 'question2' --encoding 'utf-8' --hidden_size 64 --num_layers 4 --out_features 128 --dropout 0.2 --bidirectional True
