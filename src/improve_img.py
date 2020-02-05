
import torch

def improve(image, net, epochs):
	# improve image

	image = torch.nn.Parameter(image)
	optimizer = torch.optim.Adam({image}, lr=0.001)
	loss_fn = torch.nn.MSELoss()
	pos_label = torch.tensor(1, dtype=torch.float)
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
			print("Epoch Loss: {}".format(loss))
		except:
			print("Interrupted.")
			return image
	print('\n')
	return image

def test(epochs=20):
	global net
	global image
	image = torch.randn(8)
	net = torch.nn.Linear(8,1)
	image = improve(image, net, epochs)






