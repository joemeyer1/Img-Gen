
# trains net
import torch
from tqdm import tqdm
from src.batch_data import batch as get_batches

def train_net(net, data, epochs=1000, batch_size = 100, verbose=True, lr=.001, save_best_net=True):

	# train net
	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	best_net, min_loss = None, -float('inf')
	with tqdm(range(epochs)) as epoch_counter:
		try:
			for epoch in epoch_counter:
				tot_batch_loss = 0
				batches = get_batches(data, batch_size)
				with tqdm(range(len(batches))) as batch_counter:
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
						if verbose and batch_i > 0:
							running_loss = tot_batch_loss / float(batch_i+1)
							batch_counter.desc = str(running_loss)
							batch_counter.refresh()
							# epoch_counter.write("\t Epoch {} Running Loss: {}\n".format(epoch, running_loss))
				# report loss
				avg_loss = tot_batch_loss / float(len(batches))
				epoch_counter.write(" Epoch {} Avg Loss: {}\n".format(epoch, avg_loss))
				if save_best_net and avg_loss < min_loss:
					best_net, min_loss = net.copy(), avg_loss.copy()
				epoch_counter.desc = str(avg_loss)

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