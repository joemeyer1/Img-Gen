
# trains net
import torch

def train_net(net, data, epochs=1000):

	# train net
	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	for _ in range(epochs):
		# try:
		# track epoch loss
		tot_loss = 0
		i = 0
		for features, label in data:
			features = features.reshape([1]+list(features.shape))
			label = label.reshape(1)
			# prepare for backprop
			optimizer.zero_grad()
			# compute prediction
			output = net(features)[0]
			# print("out: {}\n\tlabel: {}\n".format(output, label))
			# compute loss
			loss = loss_fn(output, label)
			# compute loss gradient
			loss.backward()
			# update weights
			optimizer.step()
			# report loss
			tot_loss += loss.item()
			if i%(len(data)//10) == 0:
				print('{}.'.format(i), end='', flush=True)
			i += 1
		# report loss
		avg_loss = tot_loss / float(len(data))
		print("Avg Epoch Loss: {}".format(avg_loss))
		# except:
		# 	print("Interrupted.")
		# 	return net
	print('\n')
	return net