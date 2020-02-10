
# trains net
import torch
from tqdm import tqdm
from src.batch_data import batch as get_batches

def train_net(net, data, epochs=1000, batch_size = 100):

	# train net
	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
	with tqdm(range(epochs)) as epoch_counter:
		for epoch in epoch_counter:
			tot_batch_loss = 0
			for batch in get_batches(data, batch_size):
				try:
					features, labels = batch
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
					tot_batch_loss += loss.item()
				except:
					print("Interrupted.")
					return net
			# report loss
			avg_loss = tot_batch_loss / float(batch_size)
			epoch_counter.write(" Epoch {} Avg Loss: {}\n".format(epoch, avg_loss))
	print('\n')
	return net