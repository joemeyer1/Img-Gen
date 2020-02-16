
import torch

def improve(image, net, epochs, verbose=True):
	# improve image

	image = torch.nn.Parameter(image)
	optimizer = torch.optim.Adam({image}, lr=10, amsgrad=True)
	loss_fn = torch.nn.MSELoss()
	pos_label = torch.tensor([[1*10000]]*len(image), dtype=torch.float)
	for _ in range(epochs):
		try:
			# prepare for backprop
			optimizer.zero_grad()
			# compute prediction
			output = net(image)
			# compute loss
			loss = loss_fn(output, pos_label)
			# compute loss gradient
			loss.backward()
			# update image
			optimizer.step()
			# report loss
			if verbose:
				print("Epoch Loss: {}".format(loss))
		except:
			print("Interrupted.")
			return image
	print('\n')
	return image

def test(epochs=20, verbose=False):
	global net
	global image
	image = torch.randn(8)
	print("image: {}".format(image))
	net = lambda x:torch.sigmoid(torch.sum(x))
	# net = torch.nn.Linear(8,1)
	print("net(image): {}".format(net(image)))
	image = improve(image, net, epochs, verbose=verbose)
	print("image: {}".format(image))
	print("net(image): {}".format(net(image)))


def test2(epochs=100, verbose=False):
	import pickle
	# global net
	# global image
	image = torch.randn(1,3,256,256)
	print("image: {}".format(image))
	with open('net.pickle','rb') as f:
		net = pickle.load(f)
	# net = torch.nn.Linear(8,1)
	print("net(image): {}".format(net(image)))
	image = improve(image, net, epochs, verbose=verbose)
	print("image: {}".format(image))
	print("net(image): {}".format(net(image)))
	return image


