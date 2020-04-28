
# pickle used for saving/loading net
import pickle
import fire
import torch
from numpy.random import shuffle
from random import randint

# import train libs
from src.cnn_classifier import CNNClassifier
# from src.classifier import Classifier as CNNClassifier
# from src.data_gen_toy import get_data
from src.data_gen import get_data
from src.train import train_net


def main(save_net_as='net-sunset-2-17.pickle', get_net_from=None, n=1000, epochs=1000, batch_size=100, img_size=(256, 256), lr=.0001):
	# pass None for get_net_from to make a new net.
	save_net_as = 'nets/'+save_net_as
	global net
	if img_size == 'hd':
		img_size = (1024, 1024)
	net = train_img_net(save_net_as, get_net_from, n, epochs, batch_size, img_size, lr)

def train_img_net(save_net_as='net-sunset-2-16.pickle', get_net_from='net-sunset.pickle', n=1000, epochs=1000, batch_size=100, img_size=(256, 256), lr=.0001):
	# get net
	print("Getting Net...")
	net = get_net(get_net_from, img_size)
	# get data [(tensor(image/non-image), tensor(P(image)), ... ]
	print("Getting Data...")
	data = get_data(n, img_size)
	# train net on data
	print("Training Net...")
	net = train_net(net, data, epochs=epochs, batch_size=batch_size, verbose=True, lr=lr)
	# save net
	with open(save_net_as, 'wb') as f:
		pickle.dump(net, f)




# HELPERS

# helper for train_img_net()
def get_net(filename, img_size):
	# get net
	if filename:
		# load from binary file
		try:
			print("Seeking Net...")
			with open(filename, 'rb') as f:
				net = pickle.load(f)
		except:
			print("Net not found. Making new one.")
			# get new net if old net not found
			net = CNNClassifier(max(img_size))
	else:
		# get new net
		net = CNNClassifier(max(img_size))

	return net



if __name__ == '__main__':
	fire.Fire(main)
