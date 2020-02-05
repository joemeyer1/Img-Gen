
# data [(tensor(image/non-image), tensor(P(image)), ... ]
import torch
from numpy.random import shuffle
from PIL import Image
import os
import sys
sys.path.append('/Users/joe/img_gen/src')

def get_data(n=100):
	pos_images = get_pos_images(n//2)
	neg_images = get_neg_images(len(pos_images))
	images = neg_images+pos_images
	shuffle(images)
	return images

# HELPERS for get_data()

def get_pos_images(n, dir_name='src/sunsets'):
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
	# return data w pos labels
	return [(img_vec, torch.tensor(1, dtype=torch.float)) for img_vec in img_vecs]

def get_neg_images(n):
	# return rand imgs w neg labels
	return [(torch.randn(3, 256, 256), torch.tensor(0, dtype=torch.float)) for _ in range(n)]


# helpers for get_pos_images()

def get_image_vec(fname):
	im = Image.open(fname).resize((256,256))
	r, g, b = get_band_lists(im)
	img_vec = get_tensor(r, g, b)
	return img_vec

# helpers for get_image_vec()
def get_band_lists(im):
	data = im.split()
	r, g, b = (list(d.getdata()) for d in data)
	return r, g, b

def get_tensor(r, g, b):
	r, g, b = torch.tensor(r, dtype=torch.float), torch.tensor(g, dtype=torch.float), torch.tensor(b, dtype=torch.float)
	rgb = torch.cat((r, g, b)).reshape(3, 256, 256)
	return rgb




