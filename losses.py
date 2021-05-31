import torch as torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
	def __init__(self, margin=0.1):
		super(TripletLoss, self).__init__()
		self.margin = margin

	def forward(self, x):
		anchor, positive, negative = torch.chunk(x, 3, dim=0)
		dp = torch.sum(torch.square(anchor - positive), axis=1)
		dn = torch.sum(torch.square(anchor - negative), axis=1)
		losses = F.relu(dp - dn + self.margin)

		# TODO: should we do mean?
		return losses.sum()


class NPairLoss(nn.Module):
	def __init__(self, num_samples_per_class):
		super(NPairLoss, self).__init__()
		self.num_samples_per_class = num_samples_per_class

	def generate_tuples(self, x):
		"""
			Input: 2N x D
			Returns
				n_anchors: torch.Tensor of size N x 1 x D
				n_positive: torch.Tensor of size N x 1 x D
				n_negatives: torch.Tensor of size N x N-1 x D
		"""
		Ntup, D = x.shape
		M = self.num_samples_per_class
		N = int(Ntup / M)
		x = x.view(N, M, D)
		# print(N, M, D)

		n_anchors, n_positive = [], []
		for i in range(N):
			anchor, positive = x[i][torch.randperm(M)[:2]]
			n_anchors.append(anchor.unsqueeze(0))
			n_positive.append(positive.unsqueeze(0))

		n_anchors = torch.cat((n_anchors), dim=0).unsqueeze(1) #N x 1 x D
		n_positive = torch.cat((n_positive), dim=0).unsqueeze(1) #N x 1 x D
		
		n_negatives = []
		for i in range(N):
			negs_i = torch.cat((x[:i, 1, :], x[i+1:, 1, :])).unsqueeze(0)
			n_negatives.append(negs_i)
		n_negatives = torch.cat((n_negatives), dim=0) # N x N-1 x D

		return n_anchors, n_positive, n_negatives


	def forward(self, x):
		# x is of size NM x D, M=2 default

		# import pdb; pdb.set_trace()
		anchors, positives, negatives = self.generate_tuples(x)
		# pdb.set_trace()

		neg_pos = negatives - positives #N x N-1 x D broadcasted
		# pdb.set_trace()
		anch_neg_pos = torch.exp(torch.matmul(anchors, neg_pos.permute(0, 2, 1)).squeeze(1)) #N x N-1
		# import pdb; pdb.set_trace()
		anch_neg_pos = torch.log(1 + anch_neg_pos.sum(dim=1)) #N
		# pdb.set_trace()

		return anch_neg_pos.sum()