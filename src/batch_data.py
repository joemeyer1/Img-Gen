
from numpy.random import shuffle
from math import ceil
import torch

def batch(data, batch_size):
	feats, labs = data
	indices = torch.randperm(len(feats))
	batches = []
	for batch_n in range(len(feats)//batch_size):
		features, labels = [], []
		batch_inds = indices[batch_n*batch_size:(batch_n+1)*batch_size]
		for i in batch_inds:
			feature, label = feats[i], labs[i]
			features.append(feature)
			labels.append(label)
		features, labels = torch.stack(features), torch.stack(labels)
		batches.append((features, labels))
	return batches




def batch_split(data, batch_size):
	feats, labs = data
	indices = torch.randperm(len(feats))
	feature_batches, label_batches = [], []
	for batch_n in range(len(feats)//batch_size):
		features, labels = [], []
		batch_inds = indices[batch_n*batch_size:(batch_n+1)*batch_size]
		for i in batch_inds:
			feature, label = feats[i], labs[i]
			features.append(feature)
			labels.append(label)
		feature_batches.append(features)
		label_batches.append(labels)
	return feature_batches, label_batches