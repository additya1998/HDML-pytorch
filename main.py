import os
import argparse
from time import gmtime, strftime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, RandomSampler

from HDML import HDML
from GoogLeNet import googlenet
from losses import TripletLoss, NPairLoss
from trainer import run_experiment
from datasets.cars196 import Cars196Dataset, Cars196TripletDataset, Cars196NPairDataset
from datasets.samplers import BalancedBatchSampler


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', action='store', type=str, required=True)

# Training Parameters
parser.add_argument('--batch_size',      		action='store', type=int,   default=40)
parser.add_argument('--test_batch_size', 		action='store', type=int,   default=128)
parser.add_argument('--batch_per_epoch', 		action='store', type=int,   default=500)
parser.add_argument('--print_frq',       		action='store', type=int,   default=100)
parser.add_argument('--max_steps',       		action='store', type=int,   default=200000)
parser.add_argument('--learning_rate',   		action='store', type=float, default=7e-5)
parser.add_argument('--num_workers',     		action='store', type=int,   default=15)
parser.add_argument('--start_step',      		action='store', type=int,   default=0)
parser.add_argument('--loss_fn',      	 		action='store', type=str,   default="npair")
parser.add_argument('--num_classes_per_batch',  action='store', type=int,   default=20)
parser.add_argument('--num_samples_per_class',  action='store', type=int,   default=2)

# Model Parameters
parser.add_argument('--embedding_size', action='store', type=int,   default=128)
parser.add_argument('--image_size',     action='store', type=int,   default=227)
parser.add_argument('--weight_decay',   action='store', type=float, default=5e-3)
# parser.add_argument('--embedding_reg',  action='store', type=float, default=3e-3)
parser.add_argument('--saved_ckpt',     action='store', type=str, default=None)

# HDML Parameters
parser.add_argument('--apply_HDML',      action='store_true')
parser.add_argument('--softmax_factor',  action='store', type=float, default=1e+4)
parser.add_argument('--beta',            action='store', type=float, default=1e+4)
parser.add_argument('--lr_generator',    action='store', type=float, default=1e-2)
parser.add_argument('--lr_softmax',      action='store', type=float, default=1e-3)
parser.add_argument('--lmbda',           action='store', type=float, default=0.5)
parser.add_argument('--alpha',           action='store', type=float, default=7)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
device_ids = [int(device_id) for device_id in device_ids]
device_ids = [x for x in range(len(device_ids))]
print('GPU configuration:', device, device_ids)
args.device = device

# setup logging
args.experiment = os.path.join('logs', args.experiment, strftime("%d_%m_%Y__%H_%M_%S", gmtime()))
writer = SummaryWriter(args.experiment)

print('==> Options:', args)
print("Using GPU:", torch.cuda.is_available())

if not os.path.exists(args.experiment): os.makedirs(args.experiment)

# setup dataloaders
args.n_classes = 98
if args.loss_fn == "triplet":
	train_dataset = Cars196TripletDataset('datasets', 'train', args.image_size)
	train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=int(1e15))
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, \
					batch_size=args.batch_size, num_workers=args.num_workers)

	test_dataset = Cars196TripletDataset('datasets', 'test', args.image_size)
	test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
elif args.loss_fn == "npair":
	train_dataset = Cars196NPairDataset('datasets', 'train', args.image_size)
	train_sampler = BalancedBatchSampler(train_dataset.df.label, args.num_classes_per_batch, args.num_samples_per_class)
	train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers)

	test_dataset = Cars196NPairDataset('datasets', 'test', args.image_size)
	test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
else:
	raise NotImplementedError


if args.loss_fn == "triplet":
	loss_fn = TripletLoss(margin=0.1)
elif args.loss_fn == "npair":
	loss_fn = NPairLoss(num_samples_per_class=args.num_samples_per_class)
else:
	raise NotImplementedError

backbone_network = googlenet(pretrained=True)
network = HDML(backbone_network, loss_fn, args)
network = network.to(args.device)

if args.saved_ckpt != None:
	fpath = os.path.join(args.saved_ckpt, 'ckpt.pth.tar')
	losses, recalls = network.load(fpath, args)
	load_dict = {'losses': losses, 'recalls': recalls}
else: load_dict = None

run_experiment(train_dataloader, test_dataloader, network, writer, load_dict, args)
