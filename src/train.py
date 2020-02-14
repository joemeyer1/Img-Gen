
# trains net
import torch
from tqdm import tqdm
from src.batch_data import batch as get_batches

def train_net(net, data, epochs=1000, batch_size = 100, verbose=True):

	# train net
	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	with tqdm(range(epochs), desc='0') as epoch_counter:
		for epoch in epoch_counter:
			tot_batch_loss = 0
			batches = get_batches(data, batch_size)
			for batch_i in range(len(batches)):
				batch = batches[batch_i]
				try:
					features, labels = batch
					# prepare for backprop
					optimizer.zero_grad()
					# compute prediction
					output = torch.sigmoid(net(features))
					# print("out: {}\n\tlabel: {}\n".format(output, labels))
					# compute loss
					loss = loss_fn(output, labels)
					# compute loss gradient
					loss.backward()
					# update weights
					optimizer.step()
					# report loss
					tot_batch_loss += loss.item()
					if verbose and batch_i > 0:
						running_loss = tot_batch_loss / (float(batch_i+1)*batch_size)
						epoch_counter.desc = str(running_loss)
						epoch_counter.refresh()
						# epoch_counter.write("\t Epoch {} Running Loss: {}\n".format(epoch, running_loss))
				except:
					print("Interrupted.")
					return net
			# report loss
			avg_loss = tot_batch_loss / float(len(batches)*batch_size)
			epoch_counter.write(" Epoch {} Avg Loss: {}\n".format(epoch, avg_loss))
	print('\n')
	return net