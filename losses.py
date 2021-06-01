import torch as torch
import torch.nn as nn
import torch.nn.functional as F

def generate_tuples_npair(x, num_samples_per_class):
	"""
		Input: 2N x D
		Returns
			n_anchors: torch.Tensor of size N x 1 x D
			n_positive: torch.Tensor of size N x 1 x D
			n_negatives: torch.Tensor of size N x N-1 x D
	"""
	Ntup, D = x.shape
	M = num_samples_per_class
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
	def __init__(self, l2_reg, num_samples_per_class):
		super(NPairLoss, self).__init__()
		self.num_samples_per_class = num_samples_per_class
		self.l2_reg = l2_reg

	def forward(self, x):
		# x is of size NM x D, M=2 default
		# N x 1 x D, N x 1 x D, N x N-1 x D
		anchors, positives, negatives = generate_tuples_npair(x, self.num_samples_per_class)

		# N
		anch_pos_l2 = torch.sqrt(torch.pow(anchors.squeeze(1) - positives.squeeze(1), 2).sum(1))
		# N x N - 1
		anch_neg_l2 = torch.sqrt(torch.pow(anchors - negatives, 2).sum(2))

		# import pdb; pdb.set_trace()

		# N x N-1 -> N -> 1
		exp_dist = torch.exp(anch_pos_l2.unsqueeze(1) - anch_neg_l2).sum(1)
		metric_loss = torch.log(1 + exp_dist).mean()

		# pdb.set_trace()
		l2_loss = (anchors.squeeze(1) ** 2 + positives.squeeze(1) ** 2).sum(1).mean()

		total_loss = metric_loss + self.l2_reg * l2_loss

		# print(metric_loss, l2_loss, total_loss)

		return total_loss