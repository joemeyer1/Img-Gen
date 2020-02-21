

from torch import nn
from src.cnn import CNN
from src.classifier import Classifier

class CNNClassifier(nn.Module):
	def __init__(self, size):
		super(CNNClassifier, self).__init__()

		self.net = nn.Sequential(
			CNN(),
			Classifier(size)
		)

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