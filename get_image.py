
# pickle used for saving/loading net
import pickle

# import gen libs
from src.improve_img import improve


def gen(net_filename='net.pickle', img_filename='image'):

	# get net
	with open(net_filename, 'rb') as f:
		net = pickle.load(f)
	# get rand noise
	noise_img = torch.randn(1, 3, 256, 256)
	# improve it
	improved_img = improve(noise_img, net)
	# save it
	with open(img_filename, 'wb') as f:
		pickle.dump(improved_img, f)




