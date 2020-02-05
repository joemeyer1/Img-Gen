
# pickle used for saving/loading net
import pickle

# import train libs
from src.cnn_classifier import CNNClassifier
from src.data_gen import get_data
from src.train import train_net


def main():
	global net
	net = train_img_net()

def train_img_net(use_old=False, filename='net.pickle'):
	# get net
	print("Getting Net...")
	net = get_net(use_old, filename)
	# get data [(tensor(image/non-image), tensor(P(image)), ... ]
	print("Getting Data...")
	data = get_data()
	# train net on data
	print("Training Net...")
	net = train_net(net, data)
	# save net
	with open(filename, 'wb') as f:
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


