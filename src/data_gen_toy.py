
# data [(tensor(image/non-image), tensor(P(image)), ... ]
import torch
from numpy.random import shuffle
from PIL import Image
import os
import sys
sys.path.append('/Users/joe/img_gen/src')

def get_data(n=1000):
	pos_images = get_pos_images(n)
	neg_images = get_neg_images(n//4, 'g') + get_neg_images(n//4, 'b') + get_neg_images(n//4, 'r', 100) + [(torch.randn(3, 256, 256), torch.tensor([0], dtype=torch.float)) for _ in range(n//4)]
	images = neg_images+pos_images
	# mix up images
	indices = [i for i in range(len(images))]
	shuffle(indices)
	features, labels = [], []
	for j in range(len(indices)):
		i = indices[j]
		feature, label = images[i]
		features.append(feature)
		labels.append(label)

	return torch.stack(features), torch.stack(labels)

# HELPERS for get_data()

def get_pos_images(n, color='r', val=255):
	return [(get_color(color, val), torch.tensor([1], dtype=torch.float))]*n

def get_neg_images(n, neg_color='g', val=255):
	# return rand imgs w neg labels
	return [(get_color(neg_color, val), torch.tensor([0], dtype=torch.float))]*n

	# [(torch.randn(3, 256, 256), torch.tensor([0], dtype=torch.float)) for _ in range(n)]


# helpers for get_pos_images()

def get_color(color='r', val=255):
	color = {'r':0, 'g':1, 'b':2}[color]
	bands = []
	for c in range(3):
		if c != color:
			bands.append(torch.zeros((256, 256), dtype=torch.float))
		else:
			bands.append(torch.ones((256, 256), dtype=torch.float)*val)

	return torch.cat(bands).reshape(3, 256, 256)


