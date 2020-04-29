# Img-Gen

Image generation in this project consists of 2 steps:


1) Train an image classifier model to distinguish positive example images from negative example images.

2) Generate a start image (i.e. uniform gray or random), then adjust it through gradient descent to make the classifier model label it as a positive example.


## Important Files

`main_train.py` : train classifier

`main_run.py` : generate image adjusted for classifier

`src/cnn_classifier.py` : defines classifier model

`src/data_gen.py` : generate training data

`src/train.py` : The actual training code wrapped by `main_train.py`


## Instructions

0) First, download this repo and add a folder called `img_data` - containing `.jpg` files of whatever you want to classify - to this repo's root directory (see `src/data_gen.py.get_pos_images()` if you want to rename it).

1) Then run `python3 main_train.py [new net_name]` (for example `python3 main_train.py net.pickle`). Once loss has stopped decreasing or you run out of patience, interrupt the loop and save the model by pressing `ctrl-c`.

2) Once a model has been trained, run `python3 main_run.py [trained net_name]` (for example `python3 main_run.py net.pickle`). Periodically, a file called `temp.jpg` will automatically be updated with the current generated image. Once loss has stopped decreasing, the generated image stops improving or you run out of patience, interrupt the loop and save the generated image by pressing `ctrl-c`.


## About

This project is in progress but should essentially work. Download it, play with it. If you want, improve it and share your work.

## To Do

Use an encoder-decoder architecture for classification. That way images could be modified/toyed with in more interesting ways, and a model could yield more variety in its generated images.





