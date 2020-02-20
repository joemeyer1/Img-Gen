
from numpy.random import shuffle
# import torch

# def batch(data, batch_size):
# 	feats, labs = data
# 	indices = torch.randperm(len(feats))
# 	batches = []
# 	for batch_n in range(len(feats)//batch_size):
# 		features, labels = [], []
# 		batch_inds = indices[batch_n*batch_size:(batch_n+1)*batch_size]
# 		for i in batch_inds:
# 			feature, label = feats[i], labs[i]
# 			features.append(feature)
# 			labels.append(label)
# 		features, labels = torch.stack(features), torch.stack(labels)
# 		batches.append((features, labels))
# 	return batches




# def batch_split(data, batch_size):
# 	feats, labs = data
# 	indices = torch.randperm(len(feats))
# 	feature_batches, label_batches = [], []
# 	for batch_n in range(len(feats)//batch_size):
# 		features, labels = [], []
# 		batch_inds = indices[batch_n*batch_size:(batch_n+1)*batch_size]
# 		for i in batch_inds:
# 			feature, label = feats[i], labs[i]
# 			features.append(feature)
# 			labels.append(label)
# 		feature_batches.append(features)
# 		label_batches.append(labels)
# 	return feature_batches, label_batches



def batch(pos_data, neg_data, batch_size):
	pos_data = partition(pos_data, batch_size//2)
	neg_data = partition(neg_data, len(pos_data))
	batches = []
	for i in range(len(pos_data)):
		batch = pos_data[i]+neg_data[i]
		shuffle(batch)
		batches.append(batch)
	return batches

def partition(data, batch_size):
	shuffle(data)
	batches = []
	i = 0
	while i+batch_size < len(data):
		batch = data[i:i+batch_size]
		batches.append(batch)
		i += batch_size
	return batches









