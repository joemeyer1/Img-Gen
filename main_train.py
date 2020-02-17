
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


def main(use_old=False, save_net_as='net-sunset-2-16.pickle', get_net_from='net-sunset.pickle'):
	# pass None for get_net_from to make a new net.
	use_old = bool(use_old)
	global net
	net = train_img_net(use_old, save_net_as, get_net_from)

def train_img_net(use_old=False, save_net_as='net-sunset-2-16.pickle', get_net_from='net-sunset.pickle'):
	# get net
	print("Getting Net...")
	net = get_net(use_old, get_net_from)
	# get data [(tensor(image/non-image), tensor(P(image)), ... ]
	print("Getting Data...")
	data = get_data(1000)
	# train net on data
	print("Training Net...")
	lr = .0001
	# if 'sunset' not in get_net_from:
	# 	lr *= 10
	net = train_net(net, data, epochs=1000, batch_size=100, verbose=True, lr=lr)
	# save net
	with open(save_net_as, 'wb') as f:
		pickle.dump(net, f)




# HELPERS

# helper for train_img_net()
def get_net(use_old, filename):
	# get net
	if use_old:
		# load from binary file
		try:
			print("Seeking Net...")
			with open(filename, 'rb') as f:
				net = pickle.load(f)
		except:
			print("Net not found. Making new one.")
			# get new net if old net not found
			net = CNNClassifier()
	else:
		# get new net
		net = CNNClassifier()

	return net



if __name__ == '__main__':
	fire.Fire(main)
