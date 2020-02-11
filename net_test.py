
from src.cnn_classifier import CNNClassifier as Net
from src.train import train_net

from numpy.random import shuffle
from random import randint

import torch

def test_net(n_data=100):
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
			features.append(torch.ones(3,256,256)*randint(0,100))
			labels.append(torch.tensor([1], dtype=torch.float))
		else:
			features.append(torch.ones(3,256,256)*randint(-100,0))
			labels.append(torch.tensor([0], dtype=torch.float))
	simple_data = (features, labels)
	print("training net...")
	net = train_net(net, simple_data, batch_size=10, verbose=True)
	return net, simple_data

net, simple_data = test_net()