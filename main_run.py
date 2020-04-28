 
# pickle used for saving/loading net
import pickle
import torch
import fire
import os
# import numpy as np
from PIL import Image

# import gen libs
from src.improve_img import improve
from src.data_gen import get_image_vec

start_img_fn = "lambda n, u_val, im_size : uniform(n, u_val, im_size)"
# start_img_fn = "lambda n, u_val : random_im(n)"
# start_img_fn = "lambda n, u_val : get_image_vecs()"

# from src.image_functions import get_image_vec
# start_img_fn = "lambda n, u_val, im_size : torch.stack([get_image_vec('valley.jpg', im_size) for _ in range(n)])"

def main(net_filename='net-sunset.pickle',
		img_filename=None,
		u_val=127,
		start_img_fn=start_img_fn,
		n=10,
		show_every=10,
		im_size = (256, 256),
		epochs=10000,
		lr=10,
		temp_name='temp'):
	# get img_filename (to save as) if not given
	if not img_filename:
		img_filename = net_filename.split('.')[0].split('/')[-1]
	# get net
	net = get_net(net_filename)
	# get fn from start img str
	start_img_fn = eval(start_img_fn)
	# get start img
	img = start_img_fn(n, int(u_val), im_size)
	print("start: {}\n\n".format(format(img[0], im_size)[:10]))
	print("start size: {}".format(img[0].shape))
	print("\tnet(old): {}".format(net(img)))
	img_vec = improve(img, net, epochs,lr=lr,verbose=True, show_every=show_every, img_fname=temp_name)
	name_i = 0
	vec_i = 0
	j = n
	while j > 0:
		if not os.path.exists(img_filename+'--'+str(name_i)+'.jpg'):
			img = format(img_vec[vec_i], im_size)
			save_img(img, img_filename+'--'+str(name_i)+'.jpg', im_size)
			j -= 1
			vec_i += 1
		name_i += 1
	print("new: {}\n\n\n\n\n".format(img[:10]))
	print("\tnet(new): {}".format(net(img_vec)))
	# return img


def format(img, size=(256, 256)):
	n = size[0]*size[1]
	r,g,b=(ch.int().flatten().tolist() for ch in img)
	return [(r[i], g[i], b[i]) for i in range(n)]

def uniform(n=1, val=127, size = (256, 256)):
	w, h = size
	return torch.ones(n, 3, w, h)*val

def random_im(n=1, size = (256, 256)):
	w, h = size
	return torch.rand(n,3,w,h)*255


def get_net(net_filename):
	with open(net_filename, 'rb') as f:
		net = pickle.load(f)
	return net


def save_img(img_vec, img_filename, size=(256,256)):
	im = Image.new('RGB', size)
	im.putdata(img_vec)
	# im.show()
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



