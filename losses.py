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

	def forward(self, embs, target, hard_forward=False, n_samples=None):

		if hard_forward:
			return self.hard_forward(embs, target, n_samples)

		twoN, emb_size = embs.size()

		anchor, positive = torch.chunk(embs.view(twoN // 2, 2, emb_size), 2, dim=1)
		anchor = anchor.squeeze(1)
		positive = positive.squeeze(1)

		batch_size = anchor.size(0)
		target = target.view(target.size(0), 1)

		target = (target == torch.transpose(target, 0, 1)).float()
		target = target / torch.sum(target, dim=1, keepdim=True).float()

		logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))

		loss_ce = cross_entropy(logit, target)
		l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

		loss = loss_ce + self.l2_reg*l2_loss*0.25

		return loss

	def hard_forward(self, embs, target, n_samples):

		anchor = embs[:n_samples, :]
		hard_pos = embs[n_samples:, :]
		target = target.view(target.size(0), 1)

		for i in range(n_samples):
			positive = hard_pos[i*n_samples: (i+1)*n_samples, :]

			target = (target == torch.transpose(target, 0, 1)).float()
			target = target / torch.sum(target, dim=1, keepdim=True).float()

			logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))

			loss_ce = cross_entropy(logit, target)
			l2_loss = torch.sum(anchor**2) / n_samples + torch.sum(positive**2) / n_samples

			loss = loss_ce + self.l2_reg*l2_loss*0.25

			if i == 0:
				all_loss = loss
			else:
				all_loss += loss

		all_loss = all_loss / n_samples

		return all_loss