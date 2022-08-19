from importlib.metadata import requires
import numpy as np
import cv2 as cv
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt

from models.vgg import Vgg16, Vgg16Experimental, Vgg19

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

# Python script for image manipulation utility functions


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')

    # converts opencv format (BGR) to RGB
    img = cv.imread(img_path)[:, :, ::-1]

    # resize section
    if target_shape is not None:
        # implicitly setting height to scalar value given
        if isinstance(target_shape, int) and target_shape != -1:
            current_width, current_height = img[:-2]
            new_height = target_shape
            new_width = int(current_width/current_height*new_height)
            img = cv.resize(img, (new_width, new_height),
                            interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to given target shape
            img = cv.resize(
                img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this needs to go after cv.resize(), otherwise cv.resize pushes values out of 0-1 range
    img = img.astype(np.float32)  # uint8 ->float32
    img /= 255.0  # get to 0-1 range

    return img


def prepare_image(img_path, target_shape, device):
    img = load_image(img_path, target_shape=target_shape)

    # normalize using ImageNet's mean and applying suitable transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(img).to(device).unsqueeze(0)

    return img


def save_image(img, img_path):
    if (len(img.shape) == 2):
        img = np.stack((img,)*3, axis=-1)
    # [:,:,::-1] converts rgb into bgr(opencv  format)
    cv.imwrite(img_path, img[:, :, ::-1])

# def generate_out_img_name


# end of image manipulation util functions

# initially it takes pytorch some time to load the model into local cache
def prepare_model(model, device):
    # we are not tuning the model weights, only optimizing the image pixels, hence requires_grad=False

    experimental = False

    if model == "vgg16":
        if experimental:
            # many more layers can be varied for style representation
            model = Vgg16Experimental(requires_grad=False, show_progress=True)
        else:
            model = Vgg16(requires_grad=False, show_progress=True)

    elif model == "vgg19":
        model = Vgg19(requires_grad=False, show_progress=True)
    else:
        raise Exception(f"{model} not supported. ")

    content_fms_index = model.content_feature_maps_index
    style_fms_indices = model.style_feature_maps_indices

    return model.to(device).eval(), content_fms_index, style_fms_indices


def gram_matrix(x, should_normalize=True):
    b, ch, h, w = x.size()
    features = x.view(b, ch, -1)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch*h*w

    return gram


""" 
Total variation loss wasn't included originally in the original NST paper. 
reducing only the style and content losses leads to highly noisy outputs and discontinuity among pixels, 
variation loss is also included in the total_loss of NST. 
This loss ensures spatial continuity and give a nice smoothness to the generated image to avoid noisy and overly pixelated results. 
This is done by finding the difference between the neighbour pixels and minimizing it
"""


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1]-y[:, :, :, 1:])) +\
        torch.sum(torch.abs(y[:, :, :-1, :]-y[:, :, 1:, :]))
