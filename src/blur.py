import numpy as np
import glob
import cv2
import scipy.io
from scipy.misc import imread, imsave
import os


def add_motion_blur(img):
    kernels = glob.glob('blur_kernels/*.png')
    idx = np.random.randint(0, len(kernels))

    kernel = imread(kernels[idx])
    kernel = kernel / np.sum(kernel)  # normalize

    blurred = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    return blurred

def save_test_images():
    mnist_test = scipy.io.loadmat('mnist_test.mat')['test']
    os.makedirs('./data/test_images/', exist_ok=True)

    for i, img in enumerate(mnist_test):
        img = np.reshape(img, newshape=(28, 28))
        imsave('./data/test_images/img_{:05d}.png'.format(i), img)


def main():
    mnist = scipy.io.loadmat('mnist.mat')['mnist_data']

    for i, img in enumerate(mnist):
        reshaped_img = np.reshape(img * 255., newshape=(28, 28))
        imsave("./data/target/img_{:05d}.png".format(2*i), reshaped_img)
        print(i)
        for j in range(3):
            blurred = add_motion_blur(reshaped_img)
            imsave("./data/blurred/img_{:05d}_{}.png".format(2*i, j), blurred)

        n = np.random.randint(1, 4)
        img_rot = np.rot90(reshaped_img, k=n)
        imsave("./data/target/img_{:05d}.png".format(2*i+1), img_rot)
        for j in range(3):
            blurred = add_motion_blur(img_rot)
            imsave("./data/blurred/img_{:05d}_{}.png".format(2*i+1, j), blurred)


if __name__ == '__main__':
    save_test_images()
    main()
