
from torch import nn

class CNN(nn.Module):
	def __init__(self, depth=3):
		super(CNN, self).__init__()

		self.net = nn.Sequential()

		for i in range(depth):
			layer = ConvBlock()
			layer_name = "block"+str(i)
			self.net.add_module(layer_name, layer)

	def forward(self, x):
		return self.net(x)


class ConvBlock(nn.Module):
	def __init__(self):
		super(ConvBlock, self).__init__()

		conv_layer = nn.Conv2d(in_channels=3,
								out_channels=3,
								kernel_size=3,
								stride=1,
								padding=1
		)

		pool_layer = nn.MaxPool2d(kernel_size=3,
									stride=1,
									padding=1
		)

		self.block = nn.Sequential(
						nn.Dropout(),
						conv_layer,
						nn.ReLU(),
						pool_layer
		)
		
	def forward(self, x):
		return self.block(x)