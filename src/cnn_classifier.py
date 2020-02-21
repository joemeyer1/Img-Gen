

from torch import nn
from math import log2
from src.cnn import CNN
from src.classifier import Classifier

class CNNClassifier(nn.Module):
	def __init__(self, size=256):
		super(CNNClassifier, self).__init__()

		self.net = nn.Sequential()

		assert size >= 256
		if size > 256:
			# round down to next highest power of 2
			size = 2**int(log2(size))
			# scaler_cnn converts 3*n*n tensors to 3*256*256
			scaler_cnn = CNN(shape=[3]*int(log2(size//256)), stride=2)
			self.net.add_module('scaler', scaler_cnn)
		# main_net does the bulk of the computation
		main_net = nn.Sequential(
			CNN(),
			Classifier()
		)
		self.net.add_module('main', main_net)

		

	def forward(self, x):
		return self.net(x)





def test():
	import torch
	global x
	x = torch.randn(2,3,256,256)
	global c
	c = CNNClassifier()
	global y
	y = c(x)