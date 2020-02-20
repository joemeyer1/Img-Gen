
# trains net
import torch
from copy import deepcopy
from tqdm import tqdm
from src.batch_data import batch as get_batches

def train_net(net, data, epochs=1000, batch_size = 100, verbose=True, lr=.001, save_best_net=True):

	# train net
	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	best_net, min_loss = None, float('inf')
	with tqdm(range(epochs)) as epoch_counter:
		try:
			tot_loss = 0.
			for epoch in epoch_counter:
				tot_batch_loss = 0
				batches = get_batches(data, batch_size)
				with tqdm(range(len(batches)), leave=False) as batch_counter:
					for batch_i in batch_counter:
						batch = batches[batch_i]
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
						if verbose:
							running_loss = tot_batch_loss / float(batch_i+1)
							batch_counter.desc = "Epoch {} Loss: {}".format(epoch, running_loss)#str(running_loss)
							# epoch_counter.write("\t Epoch {} Running Loss: {}\n".format(epoch, running_loss))
					batch_counter.close()
				# report loss
				tot_loss += tot_batch_loss
				avg_loss = tot_loss / ((epoch+1)*len(batches))
				epoch_loss = tot_batch_loss / float(len(batches))
				# epoch_counter.write("")
				epoch_counter.write(" Epoch {} Avg Loss: {}\n".format(epoch, epoch_loss))
				if save_best_net and avg_loss < min_loss:
					best_net, min_loss = deepcopy(net), deepcopy(avg_loss)
				epoch_counter.desc = "Total Loss: "+str(avg_loss)

		except:
			print("Interrupted.")
			if save_best_net:
				return best_net
			else:
				return net
	print('\n')
	if save_best_net:
		return best_net
	else:
		return net