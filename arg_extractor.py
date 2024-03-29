#adapted from the repository for the University of Edinburgh School of Informatics Machine Learning Practical course:
#https://github.com/CSTR-Edinburgh/mlpractical/tree/mlp2018-9/mlp_cluster_tutorial

import argparse
import json
import os
import sys
import GPUtil

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='number of epochs for training')
    parser.add_argument('--batch_size', nargs="?", type=int, default=8, help='Batch_size for experiment')
    parser.add_argument('--embedding_dim', type=int, help='dimension of the word embeddings used')
    parser.add_argument('--train_dataset_name', type=str, help='Dataset on which the system will train our model')
    parser.add_argument('--val_dataset_name', type=str, help='Dataset on which the system will evaluate our model')
    parser.add_argument('--data_path', type=str, help='the path to pull the trian and validation datasets from')
    parser.add_argument('--output_path', type=str, help='path for the model checkpoint file to go')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=True,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--gpu_id', type=str, default="None", help="A string indicating the gpu to use")
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=0,
                        help='Weight decay to use for Adam')
    parser.add_argument('--seed', nargs="?", type=int, default=100,
                         help='Seed to use for random number generator for experiment')
    parser.add_argument('--D', type=int, help='feature dimension')
    parser.add_argument('--train_size', type=int, help='number of observations in the train data set')
    parser.add_argument('--val_size', type=int, help='number of observations in the validation data set')
    parser.add_argument('--output_filename', type=str, 
                        help='the base name of the file for the model checkpoints to be outputted as')
    parser.add_argument('--network', type=str, 
                        help='the name of the type of network we want to use, must be one from the mynetworks.py file')
    parser.add_argument('--dataset_class', type=str, 
                        help='the name of the type of dataset we are using, must be one from the architectures.py file')
    parser.add_argument('--embedding_type', type=str, default = None, 
                        help='the name of the embedding model to be used, only relevant for non-SIF models')
    parser.add_argument('--max_tokens', type=int, default = None, 
                        help='the maximum number of tokens to include for padding purposes, only relevant for non-SIF models')
    parser.add_argument('--text_column1', type=str, default=None, 
                        help='the name of the column of text data from the first dataset in the pair of datasets to be linked, only relevant for non-SIF models')
    parser.add_argument('--text_column2', type=str, default=None, 
                        help='the name of the column of text data from the second dataset in the pair of datasets to be linked, only relevant for non-SIF models')
    parser.add_argument('--encoding', type=str, default=None, 
                        help='the type of encoding to be used when importing the raw text dataset')
    parser.add_argument('--hidden_size', type=int, default=None, 
                        help='the size of the hidden layers to be used in the LSTM encoder, only relevant for non-SIF models')
    parser.add_argument('--num_layers', type=int, default=None, 
                        help='the number of layers to be used in the LSTM encoder, only relevant for non-SIF models')
    parser.add_argument('--out_features', type=int, default=None, 
                        help='the size of the output of Feed-Forward layer of the LSTM-Siamese Net')
    parser.add_argument('--dropout', type=float, default=None, help='dropout probability to be used in the LSTM encoder')
    parser.add_argument('--bidirectional', type=bool, default=None, 
                        help='whether or not the LSTM encoder should be bidirectional, only relevant for non-SIF models')

    args = parser.parse_args()
    gpu_id = str(args.gpu_id)
     
    if gpu_id != "None":
        args.gpu_id = gpu_id

    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)

    if args.use_gpu == True:
        num_requested_gpus = len(args.gpu_id.split(","))
        num_received_gpus = len(GPUtil.getAvailable(order='first', limit=8, maxLoad=0.1,
                                             maxMemory=0.1, includeNan=False,
                                             excludeID=[], excludeUUID=[]))

        if num_requested_gpus == 1 and num_received_gpus > 1:
            print("Detected Slurm problem with GPUs, attempting automated fix")
            gpu_to_use = GPUtil.getAvailable(order='first', limit=num_received_gpus, maxLoad=0.1,
                                             maxMemory=0.1, includeNan=False,
                                             excludeID=[], excludeUUID=[])
            if len(gpu_to_use) > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use[0])
                print("Using GPU with ID", gpu_to_use[0])
            else:
                print("Not enough GPUs available, please try on another node now, or retry on this node later")
                sys.exit()

        elif num_requested_gpus > 1 and num_received_gpus > num_requested_gpus:
            print("Detected Slurm problem with GPUs, attempting automated fix")
            gpu_to_use = GPUtil.getAvailable(order='first', limit=num_received_gpus,
                                             maxLoad=0.1,
                                             maxMemory=0.1, includeNan=False,
                                             excludeID=[], excludeUUID=[])

            if len(gpu_to_use) >= num_requested_gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_idx) for gpu_idx in gpu_to_use[:num_requested_gpus])
                print("Using GPU with ID", gpu_to_use[:num_requested_gpus])
            else:
                print("Not enough GPUs available, please try on another node now, or retry on this node later")
                sys.exit()


    import torch
    args.use_cuda = torch.cuda.is_available()

    if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
        device = torch.cuda.current_device()
        print("use {} GPU(s)".format(torch.cuda.device_count()), file=sys.stderr)
    else:
        print("use CPU", file=sys.stderr)
        device = torch.device('cpu')  # sets the device to be CPU

    return args, device
