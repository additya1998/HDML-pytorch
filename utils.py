import os
import sys
import numpy as np
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter(object):

	def __init__(self, init):
		self.avg = init
		self.arr = []

	def reset(self):
		self.avg = np.mean(self.arr)
		self.arr = []

	def update(self, val, n=1):
		self.arr.append(val)


def distance_matrix(X):
    X = np.matrix(X)
    m = X.shape[0]
    t = np.matrix(np.ones([m, 1]))
    x = np.matrix(np.empty([m, 1]))
    for i in range(0, m):
        n = np.linalg.norm(X[i, :])
        x[i] = n * n
    D = x * np.transpose(t) + t * np.transpose(x) - 2 * X * np.transpose(X)
    return D


def evaluate_recall(features, labels, neighbours):
    """
    A function that calculate the recall score of a embedding
    :param features: The 2-d array of the embedding
    :param labels: The 1-d array of the label
    :param neighbours: A 1-d array contains X in Recall@X
    :return: A 1-d array of the Recall@X
    """
    dims = features.shape
    recalls = []
    D2 = distance_matrix(features)

    # set diagonal to very high number
    num = dims[0]
    D = np.sqrt(np.abs(D2))
    diagn = np.diag([float('inf') for i in range(0, D.shape[0])])
    D = D + diagn
    for i in range(0, np.shape(neighbours)[0]):
        recall_i = compute_recall_at_K(D, neighbours[i], labels, num)
        recalls.append(recall_i)
    return recalls


def compute_recall_at_K(D, K, class_ids, num):
    num_correct = 0
    for i in range(0, num):
        this_gt_class_idx = class_ids[i]
        this_row = D[i, :]
        inds = np.array(np.argsort(this_row))[0]
        knn_inds = inds[0:K]
        knn_class_inds = [class_ids[i] for i in knn_inds]
        if sum(np.in1d(knn_class_inds, this_gt_class_idx)) > 0:
            num_correct = num_correct + 1
    recall = float(num_correct)/float(num)

    # print('num_correct:', num_correct)
    # print('num:', num)
    # print("K: %d, Recall: %.3f\n" % (K, recall))
    return recall




