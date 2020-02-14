 
# pickle used for saving/loading net
import pickle
import torch
import fire
# import numpy as np
from PIL import Image

# import gen libs
from src.improve_img import improve

# start_img_fn = lambda : uniform(127)
start_img_fn = lambda n : random_im(n)

def main(epochs=500, net_filename='net.pickle', img_filename='image.jpg', u_val=127, start_img_fn=start_img_fn):
	net = get_net(net_filename)
	img = start_img_fn(10)
	print("old: {}\n\n".format(format(img[0])[:10]))
	print("\tnet(old): {}".format(net(img)))
	img_vec = improve(img, net, epochs,verbose=True)
	img = format(img_vec[0])
	save_img(img, img_filename)
	print("new: {}\n\n\n\n\n".format(img[:10]))
	print("\tnet(new): {}".format(net(img_vec)))
	# return img


def format(img, n=256):
	r,g,b=(ch.int().flatten().tolist() for ch in img)
	return [(r[i], g[i], b[i]) for i in range(n**2)]

def uniform(val=127):
	return torch.ones(1, 3, 256, 256)*val

def random_im(n=1):
	return torch.rand(n,3,256,256)*255


def get_net(net_filename):
	with open(net_filename, 'rb') as f:
		net = pickle.load(f)
	return net


def save_img(img_vec, img_filename):
	im = Image.new('RGB', (256,256))
	im.putdata(img_vec)
	im.show()
	# save it
	im.save(img_filename)


fire.Fire(main)

# def gen(epochs=500, net_filename='net.pickle', img_filename='image.jpg'):

# 	# get net
# 	with open(net_filename, 'rb') as f:
# 		net = pickle.load(f)
# 	img = uniform()
# 	# improve it
# 	improved_img = improve(img, net, epochs)
# 	# make it an image
# 	img_vec = format(improved_img[0])
# 	im = Image.fromarray(img_vec, 'RGB')
# 	im.show()
# 	# save it
# 	im.save(img_filename)

