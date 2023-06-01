

import numpy as np

#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  #depreceated for Keras > 2.9
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy import ndimage

from utils.dataset import get_set


def data_ch_shift(set_, level, image):
    img_temp = image.copy()
    if level > 0:
        ch_range = set_.ch_shift_range
        ch_a = np.random.randint(ch_range[0], ch_range[1])
        ch_b = np.random.randint(ch_range[0], ch_range[1])
        img_temp[:, :, ch_b] = image[:, :, ch_a]
        img_temp[:, :, ch_a] = image[:, :, ch_b]
    return img_temp


def data_noise(set_, noise_typ, level, image):
    row, col = image.shape
    ch = 0

    if noise_typ == "gaussian":
        level_set = np.linspace(set_.noise_gauss[0], set_.noise_gauss[1], set_.dataugm_level)
        mean = 0
        if image.max() <= 1:
            var = level_set[level]
        else:
            var = level_set[level] * 255
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        out = image + gauss
        return out

    elif noise_typ == "saltpepper":
        level_set = np.linspace(set_.noise_sp[0], set_.noise_sp[1], set_.dataugm_level)
        s_vs_p = 0.5
        ratio = level_set[level]
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(ratio * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1*255
        # Pepper mode
        num_pepper = np.ceil(ratio * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        out = np.random.poisson(image * vals) / float(vals)
        return out

    elif noise_typ == "speckle":
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        out = image + image * gauss
        return out


def data_blur(set_, blur_type, level, img):
    # 3D Operations
    if blur_type == 'gaussian':
        level_set = np.linspace(set_.blur_gauss[0], set_.blur_gauss[1], set_.dataugm_level)
        blur = ndimage.gaussian_filter(img, sigma=level_set[level])

    elif blur_type == 'median':
        level_set = np.linspace(set_.blur_median[0], set_.blur_median[1], set_.dataugm_level)
        blur = ndimage.median_filter(img, size=int(level_set[level]))
    return blur


def data_contrast(set_, level, img):
    from skimage.exposure import rescale_intensity
    if level > 9: level = 9
    if img.max() <= 1:
        level_set = [[0, 1], [0, 0.9], [0, 0.8], [0, 0.7], [0, 0.6], [0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0], [0.5, 1.0]]
    else:
        level_set = [[0, 255], [0, 230], [0, 204], [0, 178], [0, 153], [25, 255], [51, 255], [76, 255], [102, 255], [127, 255]]
    img = rescale_intensity(img, in_range=(level_set[level][0], level_set[level][1]))
    return img


def data_gamma(set_, level, img):
    from skimage import exposure
    level_set = np.linspace(set_.light_gamma[0], set_.light_gamma[1], set_.dataugm_level)
    img = exposure.adjust_gamma(img, level_set[level])
    return img


def data_transform(level, img):
    num_imgs = 10
    datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant')  # "constant", "nearest", "reflect" or "wrap"

    path = "/home/michael/Eigene_Daten/Thesis_Code/Data/01_base/Stitching_example/"
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=path + 'preview', save_prefix='img_gen', save_format='jpeg'):
        i += 1
        if i >= num_imgs:
            break


def get_augmentation_batch(set_, imgs, augmentlist):
    total = imgs.shape[0]
    for idx in range(0, total):
        # Noise 2D
        # Reference
        if augmentlist[idx, 1] == 1:
            imgs[idx, 0, :, :] = data_noise(set_, 'gaussian', augmentlist[idx, 2], imgs[idx, 0, :, :])
            imgs[idx, 1, :, :] = data_noise(set_, 'gaussian', augmentlist[idx, 2], imgs[idx, 1, :, :])
            imgs[idx, 2, :, :] = data_noise(set_, 'gaussian', augmentlist[idx, 2], imgs[idx, 2, :, :])
        if augmentlist[idx, 1] == 2:
            imgs[idx, 0, :, :] = data_noise(set_, 'saltpepper', augmentlist[idx, 2], imgs[idx, 0, :, :])
            imgs[idx, 1, :, :] = data_noise(set_, 'saltpepper', augmentlist[idx, 2], imgs[idx, 1, :, :])
            imgs[idx, 2, :, :] = data_noise(set_, 'saltpepper', augmentlist[idx, 2], imgs[idx, 2, :, :])
        if augmentlist[idx, 1] == 3:
            imgs[idx, 0, :, :] = data_blur(set_, 'gaussian', augmentlist[idx, 2], imgs[idx, 0, :, :])
            imgs[idx, 1, :, :] = data_blur(set_, 'gaussian', augmentlist[idx, 2], imgs[idx, 1, :, :])
            imgs[idx, 2, :, :] = data_blur(set_, 'gaussian', augmentlist[idx, 2], imgs[idx, 2, :, :])
        if augmentlist[idx, 1] == 4:
            imgs[idx, 0, :, :] = data_blur(set_, 'median', augmentlist[idx, 2], imgs[idx, 0, :, :])
            imgs[idx, 1, :, :] = data_blur(set_, 'median', augmentlist[idx, 2], imgs[idx, 1, :, :])
            imgs[idx, 2, :, :] = data_blur(set_, 'median', augmentlist[idx, 2], imgs[idx, 2, :, :])
        if augmentlist[idx, 1] == 5:
            imgs[idx, 0, :, :] = data_contrast(set_, augmentlist[idx, 2], imgs[idx, 0, :, :])
            imgs[idx, 1, :, :] = data_contrast(set_, augmentlist[idx, 2], imgs[idx, 1, :, :])
            imgs[idx, 2, :, :] = data_contrast(set_, augmentlist[idx, 2], imgs[idx, 2, :, :])
        if augmentlist[idx, 1] == 6:
            imgs[idx, 0, :, :] = data_gamma(set_, augmentlist[idx, 2], imgs[idx, 0, :, :])
            imgs[idx, 1, :, :] = data_gamma(set_, augmentlist[idx, 2], imgs[idx, 1, :, :])
            imgs[idx, 2, :, :] = data_gamma(set_, augmentlist[idx, 2], imgs[idx, 2, :, :])
        if augmentlist[idx, 1] == 7:
            imgs[idx, :, :, :] = data_ch_shift(set_, augmentlist[idx, 2], imgs[idx, :, :, :])

        # Moving
        if augmentlist[idx, 3] == 1:
            imgs[idx, 3, :, :] = data_noise(set_, 'gaussian', augmentlist[idx, 4], imgs[idx, 3, :, :])
            imgs[idx, 4, :, :] = data_noise(set_, 'gaussian', augmentlist[idx, 4], imgs[idx, 4, :, :])
            imgs[idx, 5, :, :] = data_noise(set_, 'gaussian', augmentlist[idx, 4], imgs[idx, 5, :, :])
        if augmentlist[idx, 3] == 2:
            imgs[idx, 3, :, :] = data_noise(set_, 'saltpepper', augmentlist[idx, 4], imgs[idx, 3, :, :])
            imgs[idx, 4, :, :] = data_noise(set_, 'saltpepper', augmentlist[idx, 4], imgs[idx, 4, :, :])
            imgs[idx, 5, :, :] = data_noise(set_, 'saltpepper', augmentlist[idx, 4], imgs[idx, 5, :, :])
        if augmentlist[idx, 3] == 3:
            imgs[idx, 3, :, :] = data_blur(set_, 'gaussian', augmentlist[idx, 4], imgs[idx, 3, :, :])
            imgs[idx, 4, :, :] = data_blur(set_, 'gaussian', augmentlist[idx, 4], imgs[idx, 4, :, :])
            imgs[idx, 5, :, :] = data_blur(set_, 'gaussian', augmentlist[idx, 4], imgs[idx, 5, :, :])
        if augmentlist[idx, 3] == 4:
            imgs[idx, 3, :, :] = data_blur(set_, 'median', augmentlist[idx, 4], imgs[idx, 3, :, :])
            imgs[idx, 4, :, :] = data_blur(set_, 'median', augmentlist[idx, 4], imgs[idx, 4, :, :])
            imgs[idx, 5, :, :] = data_blur(set_, 'median', augmentlist[idx, 4], imgs[idx, 5, :, :])
        if augmentlist[idx, 3] == 5:
            imgs[idx, 3, :, :] = data_contrast(set_, augmentlist[idx, 4], imgs[idx, 3, :, :])
            imgs[idx, 4, :, :] = data_contrast(set_, augmentlist[idx, 4], imgs[idx, 4, :, :])
            imgs[idx, 5, :, :] = data_contrast(set_, augmentlist[idx, 4], imgs[idx, 5, :, :])
        if augmentlist[idx, 3] == 6:
            imgs[idx, 3, :, :] = data_gamma(set_, augmentlist[idx, 4], imgs[idx, 3, :, :])
            imgs[idx, 4, :, :] = data_gamma(set_, augmentlist[idx, 4], imgs[idx, 4, :, :])
            imgs[idx, 5, :, :] = data_gamma(set_, augmentlist[idx, 4], imgs[idx, 5, :, :])
        if augmentlist[idx, 3] == 7:
            imgs[idx, :, :, :] = data_ch_shift(set_, augmentlist[idx, 4], imgs[idx, :, :, :])
    return imgs


def get_augmentationlist(set_, seed, no_val, selections=None, methods=None, levels=None):
    if selections is None or methods is None or levels is None:
        ratio = set_.dataugm_ratio
        augmentlist = np.zeros((no_val, 5), dtype=np.uint8)
        distribution = np.zeros((no_val), dtype=np.uint8)
        distribution[0:round(no_val*ratio)] = 1
        seed.shuffle(distribution)
        augmentlist[:, 0] = range(0, no_val)
        augmentlist[:, 1] = seed.randint(1, 8, no_val) * distribution                   # method 1
        augmentlist[:, 2] = seed.randint(0, set_.dataugm_level, no_val) * distribution  # level 1
        augmentlist[:, 3] = seed.randint(1, 8, no_val) * distribution                   # method 2
        augmentlist[:, 4] = seed.randint(0, set_.dataugm_level, no_val) * distribution  # level 2
    else:
        augmentlist = np.zeros((no_val, 6), dtype=np.uint8)
        augmentlist[0:no_val, 0] = range(0, 1)
        if selections[0] == 1:
            augmentlist[0:no_val, 1] = methods
            augmentlist[0:no_val, 2] = levels
        if selections[1] == 1:
            augmentlist[0:no_val, 3] = methods
            augmentlist[0:no_val, 4] = levels
    return augmentlist


if __name__ == '__main__':
    from main_plot import show_methods
    # Get initialization
    set_test = get_set()
    path = "/home/michael/Eigene_Daten/Thesis_Code/Data/01_base/Stitching_example/"
    img = ndimage.imread(path + 'fundus_img0_patch_c0_0.jpg', mode='RGB')
    show_methods(set_test, img)
