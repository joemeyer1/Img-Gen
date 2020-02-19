 
# pickle used for saving/loading net
import pickle
import torch
import fire
# import numpy as np
from PIL import Image

# import gen libs
from src.improve_img import improve
from src.data_gen import get_image_vec

start_img_fn = lambda n, u_val : uniform(n, u_val)
# start_img_fn = lambda n, u_val : random_im(n)
# start_img_fn = lambda n, u_val : get_image_vecs()

def main(epochs=10000, net_filename='net-sunset.pickle', img_filename='image', u_val=127, start_img_fn=start_img_fn, n=10, show_every=10):
	n = int(n)
	epochs = int(epochs)
	net = get_net(net_filename)
	img = start_img_fn(n, int(u_val))
	print("old: {}\n\n".format(format(img[0])[:10]))
	print("\tnet(old): {}".format(net(img)))
	img_vec = improve(img, net, epochs,verbose=True, show_every=show_every)
	for i in range(n):
		img = format(img_vec[i])
		save_img(img, img_filename+str(i)+'.jpg')
		if i == 0:
			print("new: {}\n\n\n\n\n".format(img[:10]))
			print("\tnet(new): {}".format(net(img_vec)))
	# return img


def format(img, n=256):
	r,g,b=(ch.int().flatten().tolist() for ch in img)
	return [(r[i], g[i], b[i]) for i in range(n**2)]

def uniform(n=1, val=127):
	return torch.ones(n, 3, 256, 256)*val

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

def get_image_vecs(n=float('inf'), dir_name='recent'):
	import os
	fnames = os.listdir(dir_name)
	img_vecs = []
	i = 0
	while n and i < len(fnames):
		fname = fnames[i]
		fpath = os.path.join(dir_name, fname)
		try:
			img_vec = get_image_vec(fpath)
			img_vecs.append(img_vec)
			n -= 1
		except:
			print("{} invalid.".format(fpath))
			# image file invalid
			pass
		i += 1
	return torch.stack(img_vecs)

if __name__ == '__main__':
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

