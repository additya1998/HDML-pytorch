import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import datasets, transforms

"""
	Taken from: https://github.com/adambielski/siamese-triplet/
"""
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


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
		super(Cars196TripletDataset, self, source_path, dataset_type, image_size).__init__()

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


if __name__ == '__main__':

	dataset = Cars196Dataset('.', 'train', 227)
	sampler = RandomSampler(dataset, replacement=True, num_samples=int(1e15))
	dataloader = DataLoader(dataset, sampler=sampler,  batch_size=128, num_workers=0)

	for step, data in enumerate(dataloader):
		print(step, data[0].shape, data[1].shape, data[2].shape, data[2])

		if step > 10: break
