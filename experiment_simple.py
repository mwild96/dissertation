import numpy as np
import arg_extractor
import architectures
import mynetworks
import torch 
import torch.nn as nn
import torch.optim as optim


args, device = arg_extractor.get_args()
np.random.RandomState(seed = args.seed)
torch.manual_seed(seed=args.seed) #not sure why we need both and what each of them do?


#parameters
epochs = args.epochs
batch_size = args.batch_size 
output_file = args.output_filename
train_size = args.train_size
val_size = args.val_size
D = args.D
#embedding_dim = args.embedding_dim
#max_tokens = args.max_tokens #we never needed this
train_path = args.data_path + args.train_dataset_name +'.csv'
val_path = args.data_path + args.val_dataset_name +'.csv'
dataset_class = args.dataset_class
network = args.network
embedding_type = args.embedding_type
weight_decay_coefficient = args.weight_decay_coefficient

#set up the data
train_loader, val_loader = architectures.data_loaders_builder(dataset_class, batch_size, embedding_type, train_path, val_path, D)


#create an instance of the network
net = eval('mynetworks.' + network + '(input_size = D)')
net = net.to(device)


#I THINK THESE TWO WILL ALWAYS BE THE SAME
#loss 
criterion = nn.BCELoss()
criterion = criterion.to(device)

#optimizer
optimizer = optim.Adam(net.parameters(), weight_decay = weight_decay_coefficient)



print('TRAINING FOR: ', args.train_dataset_name, ', ', 
      str(args.dataset_class), ', ',
      str(network), ', ',
      sep = '')


architectures.train(epochs, batch_size, train_loader, val_loader, train_size, val_size, D, optimizer, net, criterion, device, output_file, args.train_dataset_name, embedding_type, val = True)


print('VALIDATION FOR: ', args.train_dataset_name, ', ', 
      str(args.dataset_class), ', ',
      str(network), ', ',
      sep = '')

architectures.evaluate(val_loader, D, net, device, args.train_dataset_name, embedding_type)


#does this division of things make sense?


