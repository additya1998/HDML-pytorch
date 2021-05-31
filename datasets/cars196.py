import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from torchvision import datasets, transforms
from datasets.samplers import BalancedBatchSampler

class Cars196Dataset(Dataset):

	def __init__(self, source_path, dataset_type, image_size):

		if not isinstance(image_size, tuple):
			image_size = (image_size, image_size)

		self.source_path = source_path
		self.dataset_type = dataset_type
		self.image_size = image_size

		if self.dataset_type == 'train':
			path = os.path.join(self.source_path, 'cars196_train.csv')
			self.transforms = transforms.Compose([
				transforms.Resize(image_size),
				transforms.RandomCrop(image_size),
				transforms.RandomHorizontalFlip(p=0.5),
				transforms.ToTensor()
			])

		elif self.dataset_type == 'test':
			path = os.path.join(self.source_path, 'cars196_test.csv')
			self.transforms = transforms.Compose([
				transforms.Resize(image_size),
				transforms.CenterCrop(image_size),
				transforms.ToTensor()
			])

		self.df = pd.read_csv(path)
		self.df.label = self.df.label - 1

		print("Cars 196 {}: Images: {}, Classes: {}".\
			format(self.dataset_type, self.df.shape[0], self.df.label.unique().shape[0]))

		if self.dataset_type == 'train':
			labels = self.df.label.unique()
			self.label_to_index = {}
			for label in labels:
				self.label_to_index[label] = self.df[self.df.label == label].index

	def __len__(self):
		raise NotImplementedError


	def load_single(self, idx):
		path, label = self.df.iloc[idx]['name'], self.df.iloc[idx]['label']
		image = Image.open(os.path.join(self.source_path, path)).convert('RGB')
		image = self.transforms(image)
		return image, label

	def __getitem__(self, i):
		raise NotImplementedError


class Cars196TripletDataset(Cars196Dataset):
	def __init__(self, source_path, dataset_type, image_size):
		super(Cars196TripletDataset, self).__init__(source_path, dataset_type, image_size)

	def __len__(self):
		if self.dataset_type == 'train': return len(self.label_to_index)
		else: return self.df.shape[0]

	def __getitem__(self, class_idx):
		"""
			This dataloader is indexed by class id
		"""
		if self.dataset_type == 'train':
			total_images = len(self.label_to_index[class_idx])
			indexes = np.random.choice(total_images, 2)
			df_idx1, df_idx2 = indexes[0], indexes[1]
			df_idx1, df_idx2 = self.label_to_index[class_idx][df_idx1], \
						self.label_to_index[class_idx][df_idx2]

			class_nidx = np.random.choice(len(self.label_to_index) - 1)
			# Edited == to >=
			if class_nidx >= class_idx: class_nidx = class_nidx + 1

			total_images = len(self.label_to_index[class_nidx])
			df_idx3 = np.random.choice(total_images)
			df_idx3 = self.label_to_index[class_nidx][df_idx3]

			image1, label1 = self.load_single(df_idx1)
			image2, label2 = self.load_single(df_idx2)
			image3, label3 = self.load_single(df_idx3)

			assert(label1 == label2)
			assert(label1 != label3)

			return image1, label1, image2, label2, image3, label3

		else: return self.load_single(class_idx)


class Cars196NPairDataset(Cars196Dataset):
	def __init__(self, source_path, dataset_type, image_size):
		super(Cars196NPairDataset, self).__init__(source_path, dataset_type, image_size)

	def __len__(self):
		if self.dataset_type == 'train': 
			raise NotImplementedError
		else: return self.df.shape[0]

	def __getitem__(self, idx):
		"""
			This dataloader is indexed by image id
		"""
		return self.load_single(idx)


if __name__ == '__main__':

	# dataset = Cars196Dataset('.', 'train', 227)
	# sampler = RandomSampler(dataset, replacement=True, num_samples=int(1e15))
	# dataloader = DataLoader(dataset, sampler=sampler,  batch_size=128, num_workers=0)


	dataset = Cars196NPairDataset('.', 'train', 227)
	balanced_sampler = BalancedBatchSampler(dataset.df.label, 30, 2)
	dataloader = DataLoader(dataset, batch_sampler=balanced_sampler, num_workers=0)


	for step, data in enumerate(dataloader):
		# import pdb; pdb.set_trace()
		print(data)

		if step > 2: break
