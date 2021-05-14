import torch
import numpy as np
from utils import AverageMeter, evaluate_recall


def run_experiment(train_dataloader, test_dataloader, network, writer, load_dict, args):

	assert(args.max_steps % args.batch_per_epoch == 0)

	n_epoch = int(args.max_steps // args.batch_per_epoch)
	s_epoch = int(args.start_step //  args.batch_per_epoch)

	best_recall = -1

	if load_dict:
		losses, recalls = load_dict['losses'], load_dict['recalls']
		for i in range(recalls.shape[0]): best_recall = max(best_recall, recalls[i, :].mean())
		J = [AverageMeter(losses[-1][0]), AverageMeter(losses[-1][1])]
		J_dict = ['J_m', 'J_wd']
		if args.apply_HDML:
			for i in range(2, 8): J.append(AverageMeter(losses[-1][i]))
			J_dict.extend(['J_synth', 'J_metric', 'J_g', 'J_soft', 'J_recon', 'J_ce'])

	else:
		if args.apply_HDML: losses = np.array([]).reshape(0, 8)
		else: losses = np.array([]).reshape(0, 2)
		recalls = np.array([]).reshape(0, 6)

		J = [AverageMeter(1e6), AverageMeter(1e6)]
		J_dict = ['J_m', 'J_wd']
		if args.apply_HDML:
			for i in range(2, 8): J.append(AverageMeter(1e6))
			J_dict.extend(['J_synth', 'J_metric', 'J_g', 'J_soft', 'J_recon', 'J_ce'])


	t_steps = 0
	for epoch in range(s_epoch):
		for step in range(args.batch_per_epoch):
			network.scheduler_c.step()
			t_steps = t_steps + 1

		for i, k in enumerate(J_dict):
			writer.add_scalar('training/' + k, losses[epoch][i], t_steps)

		recall_xs = np.array([1, 2, 4, 8, 16, 32])
		for i, k in enumerate(recall_xs):
			writer.add_scalar('testing/Recall@' + str(k), recalls[epoch][i], t_steps)


	for epoch in range(s_epoch, n_epoch):

		print(">>>>> Training: {}".format(epoch))

		network.train()
		for step, data in enumerate(train_dataloader):

			if step >= args.batch_per_epoch: break

			iA, lA, iP, lP, iN, lN = data
			ims, labels = torch.cat((iA, iP, iN), axis=0), torch.cat((lA, lP, lN))
			ims, labels = ims.to(args.device), labels.to(args.device)
			n_samples = iA.shape[0]

			if args.apply_HDML:
				j_avg, j_g = J[0].avg, J[4].avg
			else: j_avg, j_g = None, None

			step_losses = network(ims, labels, j_avg, j_g)
			for i, k in enumerate(J_dict): J[i].update(step_losses[k], n_samples)

			t_steps = t_steps + 1

		for j in J: j.reset()
		for i, k in enumerate(J_dict): writer.add_scalar('training/' + k, J[i].avg, t_steps)
		for i, k in enumerate(J_dict): print("{}: {:.3f}".format(k, J[i].avg), end='  ')
		print('')

		epoch_losses = []
		for j in J: epoch_losses.append(j.avg)
		epoch_losses = np.array(epoch_losses)
		losses = np.vstack((losses, epoch_losses))



		print(">>>>> Testing: {}".format(epoch))

		total_embs, total_labels = np.array([]).reshape(0, args.embedding_size), np.array([])
		network.eval()
		for step, (ims, labels) in enumerate(test_dataloader):
			ims, labels = ims.to(args.device), labels.to(args.device)
			embs = network(ims, labels)
			total_embs   = np.vstack((total_embs, embs.cpu().numpy()))
			total_labels = np.hstack((total_labels, labels.cpu().numpy()))

		recall_xs = np.array([1, 2, 4, 8, 16, 32])
		epoch_recalls = evaluate_recall(total_embs, total_labels, recall_xs)
		for (k, v) in zip(recall_xs, epoch_recalls):
			writer.add_scalar('testing/Recall@' + str(k), v, t_steps)
			print("Recall@{}: {}".format(k, round(v, 2)))
		epoch_recalls = np.array(epoch_recalls)

		recalls = np.vstack((recalls, epoch_recalls))

		mean_recall = np.array(epoch_recalls).mean()

		if mean_recall > best_recall:
			best_recall = mean_recall
			network.save(args.experiment, 'best.pth.tar', losses, recalls, args)

		network.save(args.experiment, 'ckpt.pth.tar', losses, recalls, args)

