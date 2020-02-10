
from src.cnn_classifier import CNNClassifier as Net
from src.train import train_net
from numpy.random import shuffle

import torch

def test_net(n_data=100):
	net = Net()
	# get data
	indices = [1]*(n_data//2) + [0]*(n_data//2)
	shuffle(indices)
	# indices = torch.randperm((n_data//2)*2)
	features = []
	labels = []
	for i in indices:
		if i%2:
			features.append(torch.ones(3,256,256))
			labels.append(torch.tensor([1], dtype=torch.float))
		else:
			features.append(torch.zeros(3,256,256))
			labels.append(torch.tensor([0], dtype=torch.float))
	simple_data = ([features, labels])

	net = train_net(net, simple_data, batch_size=10)
	return net, simple_data

net, simple_data = test_net()