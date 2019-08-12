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

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate dissertation

#run the file with arguments
python experiment_simple.py --epochs 15 --batch_size 16 --D 300 --train_size 82414 --val_size 13933 --data_path "/home/s1834310/Dissertation/Data/AmazonGoogle/" --output_filename "/home/s1834310/Dissertation/Models/amaz_goog_fasttext_SIF" --use_gpu 'True' --gpu_id '0' --train_dataset_name "amaz_goog_train_fasttext_SIF" --val_dataset_name "amaz_goog_val_fasttext_SIF" --seed 100 --dataset_class 'SIFDataset' --network 'HighwayReluNet' --embedding_type 'fasttext' --weight_decay_coefficient 0.1 
