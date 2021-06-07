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

def cross_entropy(logits, target, size_average=True):
	if size_average:
		return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
	else:
		return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


class NPairLoss(nn.Module):
	"""the multi-class n-pair loss"""
	def __init__(self, l2_reg=0.02):
		super(NPairLoss, self).__init__()
		self.l2_reg = l2_reg

	def forward(self, embs, target):

		anchor, positive = torch.chunk(embs, 2, dim=1)

		batch_size = anchor.size(0)
		target = target.view(target.size(0), 1)

		target = (target == torch.transpose(target, 0, 1)).float()
		target = target / torch.sum(target, dim=1, keepdim=True).float()

		logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
		loss_ce = cross_entropy(logit, target)
		l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

		loss = loss_ce + self.l2_reg*l2_loss*0.25
		print(loss.item(), loss_ce.item(), l2_loss.item())

		return loss


# TODO npairforpulling