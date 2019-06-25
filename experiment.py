import arg_extractor
import architectures
import mynetworks
import torch.nn as nn
import torch.optim as optim


args, device = arg_extractor.get_args()
np.random.RandomState(seed = args.seed)
torch.manual_seed(seed=args.seed) #not sure why we need both and what each of them do?


#parameters
epochs = args.epochs
batch_size = args.batch_size 
output_filename = args.output_filename
D = args.D
#embedding_dim = args.embedding_dim
#max_tokens = args.max_tokens #we never needed this
train_path = args.data_path + args.train_dataset_name +'.csv'
val_path = args.data_path + args.val_dataset_name +'.csv'
dataset_class = args.dataset_class
network = args.network


#set up the data
train_loader, val_loader = architectures.data_loaders_builder(dataset_class, batch_size, train_path, val_path)


#create an instance of the network
net = eval('mynetworks.' + network + '(input_size = D)')
net = net.to('cuda')


#I THINK THESE TWO WILL ALWAYS BE THE SAME
#loss 
criterion = nn.BCELoss()

#optimizer
optimizer = optim.Adam(net.parameters())



print('TRAINING FOR: ', args.train_dataset_name, ', ', 
      str(args.dataset_class), ', ',
      str(network), ', ',
      sep = '')


architectures.train(epochs, train_loader, val_loader, D, optimizer, net, loss, output_filename, val = True)


print('VALIDATION FOR: ', args.train_dataset_name, ', ', 
      str(args.dataset_class), ', ',
      str(network), ', ',
      sep = '')

architectures.evaluate(val_loader, D, net)


#does this division of things make sense?


