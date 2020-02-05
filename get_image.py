
# pickle used for saving/loading net
import pickle
import torch
from PIL import Image

# import gen libs
from src.improve_img import improve


def gen(epochs=500, net_filename='net.pickle', img_filename='image'):

	# get net
	with open(net_filename, 'rb') as f:
		net = pickle.load(f)
	# get rand noise
	noise_img = torch.randn(1, 3, 256, 256)
	# improve it
	improved_img = improve(noise_img, net, epochs)
	# make it an imgae
	# try:
	img_vec = format(improved_img[0]).detach().numpy()
	# except:
	# 	return improved_img
	im = Image.fromarray(img_vec, 'RGB')
	im.show()
	# save it
	im.save(img_filename)




def format(img, n=256):
	return torch.stack(list(img[:,i,j] for i in range(n) for j in range(n))).reshape(n, n, 3)