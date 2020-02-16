#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 06:04:12 2020

@author: joe
"""

from PIL import Image

def uniform_img(color='r', x=256, y=256, data=None):
    if not data:
        data = uniform_img_data(x*y, color)
    i = Image.new('RGB', (x, y))
    i.putdata(data)
    i.show()
    
def uniform_img_data(color_map=None, color='r', n=256**2):
    if not color_map:
        color_map = {'r':0, 'g':0, 'b':0}    
        color_map[color] = 255
    colors = ('r', 'g', 'b')
    unit = tuple([color_map[col] for col in colors])
    return [unit]*n
    
def img_data(r,g,b):
    return uniform_img_data({'r':r, 'g':g, 'b':b})

def img(r,g,b, x=256, y=256):
    data = img_data(r,g,b)
    i = Image.new('RGB', (x, y))
    i.putdata(data)
#    i.show()
    return i

def interact_img():
    while True:
        r,g,b = (int(c) for c in input().split(' '))
        
f=img