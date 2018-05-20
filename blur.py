import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import scipy.io
from scipy.misc import imread, imsave
from scipy import ndimage
import h5py


def add_motion_blur(img):
    reshaped_img = np.reshape(img, newshape=(28, 28))
    kernels = glob.glob('blur_kernels/*.png')
    idx = np.random.randint(0, len(kernels))

    kernel = imread(kernels[idx])
    kernel = kernel / np.sum(kernel)  # normalize

    blurred = cv2.filter2D(img, -1, kernel)

    return blurred


def main():
    mnist = scipy.io.loadmat('mnist.mat')['mnist_data']

    for i, img in enumerate(mnist):
        reshaped_img = np.reshape(img, newshape=(28, 28))
        imsave("./data/target/img_{:05d}.png".format(i), reshaped_img)
        print(i)
        for j in range(1):
            blurred = add_motion_blur(reshaped_img)
            imsave("./data/blurred/img_{:05d}_{}.png".format(i, j), blurred)


if __name__  == '__main__':
    main()
