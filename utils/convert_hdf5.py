

import csv
import os
import h5py
import numpy as np

from utils.dataset import get_set
from utils.dataset import get_test
from utils.dataset import get_train
from utils.dataset import get_seed
from matplotlib.pyplot import imread
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom


class Datastruct(object):
    pass


def preprocess_scipy(imgs, set_, dat_type='default', reg_type='default', flow_size='default'):
    total = imgs.shape[0]

    if reg_type == 'shrink':
        # Rescale image
        if dat_type == 'doc':
            imgs_p2 = np.zeros((imgs.shape[0], 3, set_.h_out, set_.w_out), dtype=np.uint8)
            for i in range(imgs.shape[0]):
                img_temp = np.rollaxis(imgs[i, :, :, :], 0, 3)
                img_temp = resize(img_temp, [set_.h_out, set_.w_out], anti_aliasing=True)       # Removed Mode=RGB
                img_temp = np.rollaxis(img_temp, 2, 0)
                imgs_p2[i, :, :, :] = img_temp
        # Rescale label
        if dat_type == 'mask':
            imgs_p2 = np.zeros((imgs.shape[0], set_.h_out, set_.w_out), dtype=np.uint8)
            for i in range(imgs.shape[0]):
                imgs_p2[i, :, :] = resize(imgs[i, :, :], [set_.h_out, set_.w_out], anti_aliasing=True)      # Removed Mode=F
        # Rescale flow (order=0 Billinear)
        if dat_type in ('flowx', 'flowy'):
            imgs_p2 = np.zeros((imgs.shape[0], set_.h_out, set_.w_out), dtype=np.float32)
            for i in range(imgs.shape[0]):
                # Billinear interpolation of flow
                imgs_p2[i, :, :] = zoom(imgs[i, :, :], [set_.h_out / set_.h_patch, set_.w_out / set_.w_patch], output=None, order=0, mode='nearest', cval=0.0, prefilter=True)
                # Rescale flow pixel shift values to new size
                if dat_type == 'flowx':
                    imgs_p2[i, :, :] *= (set_.h_out/set_.h_patch)
                if dat_type == 'flowy':
                    imgs_p2[i, :, :] *= (set_.w_out/set_.w_patch)

    if reg_type == 'grow':
        # Rescale flow (order=0 Billinear)
        if dat_type in ('flowx', 'flowy'):
            imgs_p2 = np.zeros((imgs.shape[0], set_.h_patch, set_.w_patch), dtype=np.float32)
            for i in range(imgs.shape[0]):
                # Billinear interpolation of flow
                imgs_p2[i, :, :] = zoom(imgs[i, :, :], [set_.h_patch / set_.h_out, set_.w_patch / set_.w_out], output=None, order=0, mode='nearest', cval=0.0, prefilter=True)
                # Rescale flow pixel shift values to new size
                if dat_type == 'flowx':
                    imgs_p2[i, :, :] *= (set_.h_patch/set_.h_out)
                if dat_type == 'flowy':
                    imgs_p2[i, :, :] *= (set_.w_patch/set_.w_out)

    if reg_type == 'manual':
        # Rescale flow (order=0 Billinear)
        if dat_type in ('flowx', 'flowy'):
            imgs_p2 = np.zeros((imgs.shape[0], flow_size, flow_size), dtype=np.float32)
            for i in range(imgs.shape[0]):
                # Billinear interpolation of flow
                imgs_p2[i, :, :] = zoom(imgs[i, :, :], [flow_size / set_.h_out, flow_size / set_.w_out], output=None, order=0, mode='nearest', cval=0.0, prefilter=True)
                # Rescale flow pixel shift values to new size
                if set_.flow_scaled == 'pixel':
                    # Rescale flow pixel shift values to new size
                    if dat_type == 'flowx':
                        imgs_p2[i, :, :] *= (set_.h_patch/set_.h_out)
                    if dat_type == 'flowy':
                        imgs_p2[i, :, :] *= (set_.w_patch/set_.w_out)
    return imgs_p2


def check_hdf5(filename, set):
    f = h5py.File(filename, 'a')
    dset1 = f['img1']
    no_files = dset1.shape[0]
    return no_files


def rd_hdf5_batch(filename, set_, scaling, list_s, readtype):
    # Check data
    f = h5py.File(filename, 'a')
    dset1 = f['img1']
    dset2 = f['img2']
    dset3 = f['label']
    dset4 = f['img_id']
    # Check for Flow Data (backwards compatibility)
    try:
        dset5 = f['flowx']
        dset6 = f['flowy']
        flag_flow = 1
    except KeyError:
        flag_flow = 0
        flowxy = []
        print('No Flow Data in hdf5 File found')
    # Get readtype
    if readtype == 'all':
        no_slices = dset4.shape[1]
        slice_start = 0
        slice_stop = no_slices
        list_flag = 0
    if readtype == 'range':
        slice_start = list_s[0]
        slice_stop = list_s[1]
        no_slices = slice_stop - slice_start
        list_flag = 0
    if readtype == 'list':
        no_slices = len(list_s)
        list_flag = 1

    # Get scaling settings for output reshaping
    if scaling == 'p':  # resized
        h = set_.h_out
        w = set_.w_out
    else:  # orig size
        h = set_.h_patch
        w = set_.w_patch

    # Read range of slices (1D)
    if list_flag == 0:
        imgs1_1d = dset1[slice_start:slice_stop, :]
        imgs2_1d = dset2[slice_start:slice_stop, :]
        label_1d = dset3[slice_start:slice_stop, :]
        imgs_id = dset4[:, slice_start:slice_stop]
        if flag_flow == 1:
            flowx_1d = dset5[slice_start:slice_stop, :]
            flowy_1d = dset6[slice_start:slice_stop, :]

    # Read list of slices (1D)
    if list_flag == 1:
        imgs1_1d = np.ndarray((no_slices, h*w*3), dtype=np.uint8)
        imgs2_1d = np.ndarray((no_slices, h*w*3), dtype=np.uint8)
        label_1d = np.ndarray((no_slices, h*w), dtype=np.uint8)
        imgs_id = np.ndarray((no_slices), dtype=np.uint8)
        flowx_1d = np.ndarray((no_slices, h*w), dtype=np.float32)
        flowy_1d = np.ndarray((no_slices, h*w), dtype=np.float32)
        for idx, slice in enumerate(list_s):
            if not idx % 100:
                print('Read shuffled data:', idx, ' /', len(list_s))
            imgs1_1d[idx, :] = dset1[slice, :]
            imgs2_1d[idx, :] = dset2[slice, :]
            label_1d[idx, :] = dset3[slice, :]
            imgs_id[idx] = dset4[0, slice]
            if flag_flow == 1:
                flowx_1d[idx, :] = dset5[slice, :]
                flowy_1d[idx, :] = dset6[slice, :]

    # Reshape doc outputs (2D)
    total = imgs1_1d.shape[0]
    if total < no_slices:
        print('more slices selected than existing!')
    imgs = np.ndarray((total, 6, h, w), dtype=np.uint8)
    imgs[:, 0:3, :, :] = np.reshape(imgs1_1d, (no_slices, 3, h, w))
    imgs[:, 3:6, :, :] = np.reshape(imgs2_1d, (no_slices, 3, h, w))
    imgs_mask = np.ndarray((total, 1, h, w), dtype=np.uint8)
    imgs_mask[:, 0, :, :] = np.reshape(label_1d, (no_slices, h, w))

    # Reshape optical flow outputs (2D)
    if flag_flow == 1:
        flowxy = np.ndarray((total, 2, h, w), dtype=np.float32)  # int16
        flowxy[:, 0, :, :] = np.reshape(flowx_1d, (no_slices, h, w))
        flowxy[:, 1, :, :] = np.reshape(flowy_1d, (no_slices, h, w))

    return imgs, imgs_mask, imgs_id, flowxy


def wr_hdf5(dat, filename_, set_, img1, img2, label, img_id, flowx, flowy, blk, total, start, stop):
    filename = os.path.join(dat.path_dataraw, filename_)
    no_files = total
    blk_size = img1.shape[0]
    ch = img1.shape[1]
    rows = img1.shape[2]
    cols = img1.shape[3]

    img1_1d = np.reshape(img1, (blk_size, ch * rows * cols))
    img2_1d = np.reshape(img2, (blk_size, ch * rows * cols))
    label_1d = np.reshape(label, (blk_size, rows * cols))
    flowx_1d = np.reshape(flowx, (blk_size, rows * cols))
    flowy_1d = np.reshape(flowy, (blk_size, rows * cols))

    # Create HDF5 file and Data groups (initial overwrite)
    if blk == 0:
        f = h5py.File(filename, 'w')
        dset0 = f.create_dataset('header_info', (1, no_files), dtype='i8', maxshape=(None, None), compression='gzip', compression_opts=2)  # 'lzf' not readable hdf5view
        dset1 = f.create_dataset('img1', (no_files, ch * rows * cols), dtype='i8', maxshape=(None, None), compression='gzip', compression_opts=2)
        dset2 = f.create_dataset('img2', (no_files, ch * rows * cols), dtype='i8', maxshape=(None, None), compression='gzip', compression_opts=2)
        dset3 = f.create_dataset('label', (no_files, rows * cols), dtype='i8', maxshape=(None, None), compression='gzip', compression_opts=2)
        dset4 = f.create_dataset('img_id', (1, no_files), dtype='i8', maxshape=(None, None), compression='gzip', compression_opts=2)
        dset5 = f.create_dataset('flowx', (no_files, rows * cols), dtype='f', maxshape=(None, None), compression='gzip', compression_opts=2)  # i8
        dset6 = f.create_dataset('flowy', (no_files, rows * cols), dtype='f', maxshape=(None, None), compression='gzip', compression_opts=2)  # i8
    # Read HDF5 groups and append Data
    else:
        f = h5py.File(filename, 'a')
        dset1 = f['img1']
        dset2 = f['img2']
        dset3 = f['label']
        dset4 = f['img_id']
        dset5 = f['flowx']
        dset6 = f['flowy']
    dset1[start:stop, :] = img1_1d
    dset2[start:stop, :] = img2_1d
    dset3[start:stop, :] = label_1d
    dset4[start:stop, :] = img_id
    dset5[start:stop, :] = flowx_1d
    dset6[start:stop, :] = flowy_1d
    # Close file stream
    f.close()


def cr_hdf5(csv_list, dat, sett):
    print('-' * 30)
    print('Creating train images...')
    print('-' * 30)

    total = int(len(csv_list))

    # Block reading
    blk_size = sett.blk_size
    blk_no = int(np.floor(total / blk_size))
    blk_modulo = total - (blk_no * blk_size)
    blk_set = range(0, blk_no)

    if blk_modulo == 0:
        flag_mod = 0
    else:
        flag_mod = 1
        blk_set = np.append(blk_set, 1)  # add 1 element, to calculate modulo blk

    for blk_id, blk in enumerate(blk_set):
        # Check for last blk
        if flag_mod == 1 and (blk_id + 1) == blk_set.shape[0]:
            start = blk_id * blk_size
            stop = start + blk_modulo
            blk_size = blk_modulo
        else:
            start = blk_id * blk_size
            stop = start + blk_size

        # Init arrays (limited to blk size, safe memory)
        imgs1_blk = np.zeros((blk_size, 3, sett.h_patch, sett.w_patch), dtype=np.uint8)
        imgs2_blk = np.zeros((blk_size, 3, sett.h_patch, sett.w_patch), dtype=np.uint8)
        label_blk = np.zeros((blk_size, sett.h_patch, sett.w_patch), dtype=np.uint8)
        flowx_blk = np.zeros((blk_size, sett.h_patch, sett.w_patch), dtype=np.float32)  # int16
        flowy_blk = np.zeros((blk_size, sett.h_patch, sett.w_patch), dtype=np.float32)  # int16

        # Read Data to arrays
        csv_off = 0
        for idx in range(start, stop):
            img1 = imread(os.path.join(dat.path_datas, csv_list[idx + csv_off][1]), format=None)
            img1 = np.array([img1])
            img1 = np.rollaxis(img1, 3, 1)
            img2 = imread(os.path.join(dat.path_datas, csv_list[idx + csv_off][2]), format=None)
            img2 = np.array([img2])
            img2 = np.rollaxis(img2, 3, 1)
            msk = imread(os.path.join(dat.path_datas, csv_list[idx + csv_off][3]), format=None)
            msk = np.array([msk])
            if sett.flow_generated == 1:
                mskflow = np.load(os.path.join(dat.path_datas, csv_list[idx + csv_off][4]))
                mskflow = np.array([mskflow])

            # Write blk array for HDF5 export
            imgs1_blk[idx - start] = img1
            imgs2_blk[idx - start] = img2
            label_blk[idx - start] = msk
            if sett.flow_generated == 1:
                flowx_blk[idx - start] = mskflow[0, :, :, 0]
                flowy_blk[idx - start] = mskflow[0, :, :, 1]

            if idx % 100 == 0:
                print('Done: {0}/{1} images'.format(idx, total))

        # Create Indexing array
        imgs_id = range(0, total)

        # Data pre-processing
        imgs1_p = preprocess_scipy(imgs1_blk, sett, dat_type='doc', reg_type='shrink')
        imgs2_p = preprocess_scipy(imgs2_blk, sett, dat_type='doc', reg_type='shrink')
        label_p = preprocess_scipy(label_blk, sett, dat_type='mask', reg_type='shrink')
        flowx_p = preprocess_scipy(flowx_blk, sett, dat_type='flowx', reg_type='shrink')
        flowy_p = preprocess_scipy(flowy_blk, sett, dat_type='flowy', reg_type='shrink')

        # Write HDF5 File sequentially
        print('Saving files in progress...')
        if sett.cr_hdf5 == 1:
            wr_hdf5(dat, dat.path_hdf5, sett, imgs1_blk, imgs2_blk, label_blk, imgs_id, flowx_blk, flowy_blk, blk_id, total, start, stop)
        if sett.cr_hdf5_scale == 1:
            wr_hdf5(dat, dat.path_hdf5_scale, sett, imgs1_p, imgs2_p, label_p, imgs_id, flowx_p, flowy_p, blk_id, total, start, stop)
    print('Saving files done.')


def ld_csv(dat, path=False, selection='default'):
    csv_list = []
    if path == True:
        if selection == 'label':
            csvfile = open(dat.path_dataraw + dat.path_csv_label, 'rt')
        if selection == 'patchlist':
            csvfile = open(dat.path_dataraw + dat.path_patchlist, 'rt')
    else:
        if selection == 'label':
            csvfile = open(dat.path_csv_label, 'rt')
        if selection == 'patchlist':
            csvfile = open(dat.path_patchlist, 'rt')
    for row in csv.reader(csvfile, delimiter=','):
        csv_list.append(row)
    return csv_list


def main():
    # Get initialization
    dat_train = get_train()
    dat_test = get_test()
    setting = get_set()

    # Train
    if setting.cr_train == 1:
        csv_train = ld_csv(dat_train, path=True, selection='label')
        # delete csv header row
        del csv_train[0]
        if setting.shuffle_train == 1:
            seedx = get_seed(setting.seed1)
            seedx.shuffle(csv_train)
        cr_hdf5(csv_train, dat_train, setting)
    # Test
    if setting.cr_test == 1:
        csv_test = ld_csv(dat_test, path=True, selection='label')
        # delete csv header row
        del csv_test[0]
        if setting.shuffle_test == 1:
            seedx = get_seed(setting.seed1)
            seedx.shuffle(csv_test)
        cr_hdf5(csv_test, dat_test, setting)

    print('-' * 30)
    print("Path:    ", dat_train.path_dataraw)
    print('-' * 30)


if __name__ == '__main__':
    main()
