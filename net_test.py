
# from src.cnn_classifier import CNNClassifier as Net
from src.classifier import Classifier as Net
from src.train import train_net

from numpy.random import shuffle, choice
from random import randint

import torch

def test_net(n_data=1000, val_n = 100):
	print("getting net...")
	net = Net()
	print("getting data...")
	indices = [1]*(n_data//2) + [0]*(n_data//2)
	shuffle(indices)
	# indices = torch.randperm((n_data//2)*2)
	features = []
	labels = []
	for i in indices:
		if i%2:
			features.append(torch.ones(3,256,256)*randint(0,255))
			labels.append(torch.tensor([1], dtype=torch.float))
		else:
			features.append(torch.ones(3,256,256)*randint(-254,0))
			labels.append(torch.tensor([0], dtype=torch.float))
	simple_data = (features, labels)
	print("training net...")
	net = train_net(net, simple_data, batch_size=100, verbose=True)
	print("calculating confusion matrix...")
	test_net_help(net(torch.stack(simple_data[0][:val_n])), simple_data[1][:val_n])
	return net, simple_data

def test_net_help(preds, labels):
	tp, tn, fp, fn = 0,0,0,0
	for i in range(len(preds)):
		if preds[i] > .5:
			if labels[i] > .5:
				tp += 1
			else:
				fp += 1
		else:
			if labels[i] <= .5:
				tn += 1
			else:
				fn += 1
	print("tp: {}\ntn: {}\nfp: {}\nfn: {}\n\tAcc: {}\n\tPrec: {}".format(tp,tn,fp,fn,float(tp+tn)/(tp+tn+fp+fn), float(tp)/(tp+fn)))


net, simple_data = test_net()