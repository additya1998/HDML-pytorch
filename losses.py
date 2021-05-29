import torch as torch
import torch.nn as nn

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
	def __init__(self):
		super(NPairLoss, self).__init__()
		
	def generate_tuples(self, x):
		"""
			Returns
				n_anchors: torch.Tensor of size N x 1 x D
				n_positive: torch.Tensor of size N x 1 x D
				n_negatives: torch.Tensor of size N x N-1 x D
		"""
		N, M, D = x.shape
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
		# x is of size N x M, M=2 default

		anchors, positives, negatives = self.generate_tuples(x)

		neg_pos = negatives - positives #N x N-1 x D broadcasted
		anch_neg_pos = torch.exp(torch.matmul(n_anchors, n_negatives.permute(0, 2, 1)).squeeze(1)) #N x N-1
		anch_neg_pos = torch.log(1 + anch_neg_pos.sum(dim=1)) #N

		return anch_neg_pos.mean()