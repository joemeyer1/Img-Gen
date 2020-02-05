
# trains net
import torch
from tqdm import tqdm

def train_net(net, data, epochs=1000):

	# train net
	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	with tqdm(total=len(data)) as t:
		for i in range(epochs):
			try:
				# track epoch loss
				tot_loss = 0
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
				tot_loss += loss.item()
				t.update()
				# report loss
				avg_loss = tot_loss / float(len(data))
				print("Epoch {} Avg Loss: {}\n".format(i, avg_loss))
			except:
				print("Interrupted.")
				return net
			t.reset()
	print('\n')
	return net