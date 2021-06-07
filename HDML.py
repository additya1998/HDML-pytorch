import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from losses import generate_tuples_npair


def glorot_uniform_initializer(m):
	if isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.0)

def distance(embsA, embsB):
	return torch.sqrt(torch.sum(torch.square(embsA - embsB), axis=1))

class Pulling:

	def __init__(self, alpha, embedding_size):
		self.alpha = alpha
		self.embedding_size = embedding_size

	def transform(self, embs, j):
		raise NotImplementedError

class TripletPulling(Pulling):

	def __init__(self, alpha, embedding_size):
		super(TripletPulling, self).__init__(alpha, embedding_size)

	def transform(self, embs, j):
		embs_a, embs_p, embs_n = torch.chunk(embs, 3, dim=0)
		dp = distance(embs_a, embs_p) #N
		dn = distance(embs_a, embs_n) #N

		r = ((dp + (dn - dp) * np.exp(-self.alpha / j)) / dn).unsqueeze(-1).repeat(1, self.embedding_size)
		embs_nn = embs_a + torch.mul((embs_n - embs_a), r)
		mask = torch.ge(dp, dn)
		mask_inverse = ~mask
		mask = mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
		mask_inverse = mask_inverse.float().unsqueeze(-1).repeat(1, self.embedding_size)
		embs_n_ret = torch.mul(embs_n, mask) + torch.mul(embs_nn, mask_inverse)
		return torch.cat((embs_a, embs_p, embs_n_ret), axis=0)


class NPairPulling(Pulling):
	# TODO
	def __init__(self, alpha, embedding_size, num_samples_per_class):
		super(NPairPulling, self).__init__(alpha, embedding_size)
		self.num_samples_per_class = num_samples_per_class

	def transform(self, embs, j):
		embs_a, embs_p, embs_n = generate_tuples_npair(embs, self.num_samples_per_class)

		# N x 1
		dp = torch.sqrt(torch.pow(embs_a.squeeze(1) - embs_p.squeeze(1), 2).sum(1)).unsqueeze(1)
		# N x N - 1
		dn = torch.sqrt(torch.pow(embs_a - embs_n, 2).sum(2))

		r = ((dp + (dn - dp) * np.exp(-self.alpha / j)) / dn).unsqueeze(-1).repeat(1, 1, self.embedding_size)
		embs_nn = embs_a + torch.mul((embs_n - embs_a), r)
		mask = torch.ge(dp, dn)
		mask_inverse = ~mask

		mask = mask.float().unsqueeze(-1).repeat(1, 1, self.embedding_size)
		mask_inverse = mask_inverse.float().unsqueeze(-1).repeat(1, 1, self.embedding_size)
		embs_n_ret = torch.mul(embs_n, mask) + torch.mul(embs_nn, mask_inverse)

		return embs_a, embs_p, embs_n_ret



class HDML(nn.Module):

	def __init__(self, backbone_network, loss_fn, args):
		super(HDML, self).__init__()

		self.apply_HDML = args.apply_HDML
		self.backbone_network = backbone_network
		self.z_embedding_size = args.embedding_size
		self.y_embedding_size = backbone_network.embedding_size
		self.n_classes = args.n_classes
		self.metric_loss_fn = loss_fn
		self.weight_decay = args.weight_decay

		self.yz_network = nn.Sequential(
			nn.BatchNorm1d(self.y_embedding_size),
			nn.Linear(self.y_embedding_size, self.z_embedding_size),
		)
		self.yz_network.apply(glorot_uniform_initializer)

		params_c = list(self.backbone_network.parameters()) + list(self.yz_network.parameters())
		self.optimizer_c = optim.Adam(params_c, lr=args.learning_rate, weight_decay=args.weight_decay)
		self.scheduler_c = optim.lr_scheduler.MultiStepLR(self.optimizer_c, milestones=[5632, 6848], gamma=0.5)

		if self.apply_HDML:

			# constants
			self.beta, self.lmbda, self.alpha = args.beta, args.lmbda, args.alpha
			self.softmax_factor = args.softmax_factor

			# netwoks and optimizers
			if args.loss_fn == "triplet":
				self.pulling = TripletPulling(self.alpha, self.z_embedding_size)
			elif args.loss_fn == "npair":
				self.pulling = NPairPulling(self.alpha, self.z_embedding_size, args.num_samples_per_class)

			self.ce_loss_fn = nn.CrossEntropyLoss()
			self.generator = nn.Sequential(
				nn.Linear(self.z_embedding_size, 512),
				nn.ReLU(),
				nn.BatchNorm1d(512),
				nn.Linear(512, self.y_embedding_size)
			)
			self.softmax_classifier = nn.Sequential(
				nn.Linear(self.y_embedding_size, self.n_classes)
			)

			# optimizers
			self.optimizer_g = optim.Adam(self.generator.parameters(), lr=args.lr_generator)
			self.optimizer_s = optim.Adam(self.softmax_classifier.parameters(), lr=args.lr_softmax)


	def save(self, dir_path, fname, losses, recalls, args):
		state = {'net_state_dict': self.state_dict()}
		state['recalls'] = recalls
		state['losses'] = losses
		state.update(vars(args))
		save_path = os.path.join(dir_path, fname)
		torch.save(state, save_path)

	def load(self, fpath, args):
		saved = torch.load(fpath)
		exceptions = ['experiment', 'test_batch_size', 'saved_ckpt', 'start_step', 'max_steps', 'num_workers']
		for k, v in saved.items():
			if k in args and k not in exceptions:
				if getattr(args, k) != saved[k]: print("{} not same!".format(k))
				assert(getattr(args, k) == saved[k])
		state_dict = saved['net_state_dict']
		self.load_state_dict(state_dict)
		return saved['losses'], saved['recalls']


	def forward(self, x, labels, j_avg=None, j_g=None):

		n_samples = x.shape[0] // 3 # n_samples == n_classes

		# testing
		if not self.training:
			with torch.set_grad_enabled(False):
				embs_y = self.backbone_network(x)
				embs_z = self.yz_network(embs_y)
				return embs_z

		# training baseline
		if not self.apply_HDML:
			with torch.set_grad_enabled(True):
				embs_y = self.backbone_network(x)
				embs_z = self.yz_network(embs_y)
				J_m = self.metric_loss_fn(embs_z, labels)
				self.optimizer_c.zero_grad()
				J_m.backward()
				self.optimizer_c.step()

				J_wd = 0
				for param in self.backbone_network.parameters(): J_wd = J_wd + torch.sum(param ** 2)
				for param in self.yz_network.parameters(): J_wd = J_wd + torch.sum(param ** 2)
				J_wd = J_wd * self.weight_decay

			return {'J_m': J_m.item(), 'J_wd': J_wd.item()}

		if self.apply_HDML:

			# train backbone network + yz network using real samples: only compute loss
			with torch.set_grad_enabled(True):
				embs_y = self.backbone_network(x)
				embs_z = self.yz_network(embs_y)
				
				J_m = self.metric_loss_fn(embs_z)
				J_m = J_m * np.exp(-self.beta / j_g)
				self.optimizer_c.zero_grad()
				J_m.backward()

				J_wd = 0
				for param in self.backbone_network.parameters(): J_wd = J_wd + torch.sum(param ** 2)
				for param in self.yz_network.parameters(): J_wd = J_wd + torch.sum(param ** 2)
				J_wd = J_wd * self.weight_decay

			embs_y, embs_z = embs_y.detach(), embs_z.detach()

			# train softmax classifier
			with torch.set_grad_enabled(True):
				ce_preds = self.softmax_classifier(embs_y)
				J_ce = self.ce_loss_fn(ce_preds, labels)
				self.optimizer_s.zero_grad()
				J_ce.backward()
				self.optimizer_s.step()

			embs_z_hard = self.pulling.transform(embs_z, j_avg)

			# TODO: modify for npair
			embs_z_total = torch.cat((embs_z, embs_z_hard), axis=0)

			# train generator
			with torch.set_grad_enabled(True):
				embs_yg_total = self.generator(embs_z_total)
				preds_yg_total = self.softmax_classifier(embs_yg_total)

				# TODO: modify for npair
				labels_yg_total = torch.cat((labels, labels))
				J_soft = self.ce_loss_fn(preds_yg_total, labels_yg_total)
				J_soft = self.softmax_factor * self.lmbda * J_soft

				embs_yg, embs_yg_hard = torch.split(embs_yg_total, 3 * n_samples)
				J_recon = torch.sum(torch.square(embs_y - embs_yg))
				J_recon = (1 - self.lmbda) * J_recon

				J_g = J_soft + J_recon
				self.optimizer_g.zero_grad()
				J_g.backward()
				self.optimizer_g.step()

			embs_yg_hard = embs_yg_hard.detach()

			# train yz_network and (backbone + yz_network) optimizer step
			with torch.set_grad_enabled(True):
				# TODO: modify for npair, check for each a, p if you take all negs (not ment in paper)
				embs_zg_hard = self.yz_network(embs_yg_hard)
				J_synth = self.metric_loss_fn(embs_zg_hard)
				J_synth = (1 - np.exp(-self.beta / j_g)) * J_synth
				J_synth.backward()
				self.optimizer_c.step()

			J_metric = J_m + J_synth

			loss_dict = {'J_m': J_m.item(), 'J_synth': J_synth.item(), 'J_metric': J_metric.item(),
					'J_g': J_g.item(), 'J_soft': J_soft.item(), 'J_recon': J_recon.item(),
					'J_ce': J_ce.item(), 'J_wd': J_wd.item()}

			self.scheduler_c.step()

			return loss_dict
