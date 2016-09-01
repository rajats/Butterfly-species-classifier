# https://github.com/Lasagne/Recipes
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt

from lasagne.utils import floatX

def prep_image(fn, ext='jpg', IMAGE_MEAN =None):
    im = plt.imread(fn, ext)

    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # discard alpha channel if present
    im = im[:3]

    # Convert to BGR
    im = im[::-1, :, :]

    if IMAGE_MEAN is not None:
        im = im - IMAGE_MEAN

    return rawim, floatX(im[np.newaxis])
