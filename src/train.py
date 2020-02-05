
# trains net
import torch
from tqdm import tqdm

def train_net(net, data, epochs=1000):

	# train net
	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	with tqdm(range(epochs)) as t:
		for i in t:
			try:
				features, labels = data
				# prepare for backprop
				optimizer.zero_grad()
				# compute prediction
				output = net(features)
				# print("out: {}\n\tlabel: {}\n".format(output, label))
				# compute loss
				loss = loss_fn(output, labels)
				# compute loss gradient
				loss.backward()
				# update weights
				optimizer.step()
				# report loss
				avg_loss = loss.item() / float(len(features))
				t.write(" Epoch {} Avg Loss: {}\n".format(i, avg_loss))
			except:
				print("Interrupted.")
				return net
	print('\n')
	return net