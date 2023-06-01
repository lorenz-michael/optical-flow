

import os
import csv
import shutil
import time
import numpy as np

from numpy.random import RandomState
from skimage.transform import resize

from settings import get_train, get_test, get_set


def init_folders(dat_, set_):
    # Remove old train/test data Folders and recreate them
    if os.path.exists(dat_.path_datas):
        shutil.rmtree(dat_.path_datas)
    if not os.path.exists(dat_.path_datas):
        os.makedirs(dat_.path_datas)
    if os.path.exists(dat_.path_info):
        shutil.rmtree(dat_.path_info)
    # Remove files in old sample folder
    filelist = os.listdir(dat_.path_datas)
    for fileName in filelist:
        os.remove(dat_.path_datas + fileName)
    # Recreate info Folder
    if not os.path.exists(dat_.path_info) and set_.cr_mark or set_.cr_orig:
        os.makedirs(dat_.path_info)
    return 0


def get_files(dat_, set_, seed_):
    # Open files in Data folder (sorted) and shuffle if selected
    files = os.listdir(dat_.path_datao)
    files = sorted(files, key=lambda x: x.rsplit('.', 1)[0])
    if set_.datashuffle == 1:
        seed_.shuffle(files)
    return files


def get_seed(seed_init):
    return RandomState(seed_init)


def get_rescale(imgs, set_, arg):
    h, w, col = imgs.shape
    if set_.h_rescale > 1 or set_.w_rescale > 1:
        h_out = h * set_.h_rescale
        w_out = w * set_.w_rescale
        imgs = resize(imgs, [h_out, w_out], interp='bilinear')  # bilinear, nearest, cubic, bicubic,
    return imgs


def get_ovarea(lab_):
    tot = lab_.size
    nz = np.count_nonzero(lab_)
    ov = nz / tot
    return ov


def get_patchlist(dat_, set_, files_, seedx_, seedy_):
    # Patchlist =
    # [0]img_no [1]centroid_no [2]patch_no [3]height [4]width [5]x1 [6]x2 [7]y1 [8]y2 [9]Tx [10]Ty [11]Rot [12]Scale
    import cv2
    print('Create Patchlist...')
    tot_patches = len(files_) * set_.tot_centroid * (set_.no_patches)
    patches = np.zeros((tot_patches, 14), dtype=int)
    tx = 0
    ty = 0
    rot = 0
    scale = 0
    shear = 0

    for img_no in range(0, len(files_)):
        # Read data
        img = cv2.imread(dat_.path_datao + files_[img_no])
        # Rescaling (used for pyramid scaling of features later)
        img = get_rescale(img, set_, '')
        # Get doc size
        h, w = img.shape[:2]
        # Check shuffling of overlapping matrix for every doc
        ov_shuf = set_.ov_batch.copy()

        # Get patch coord. per Centroid
        for c in range(0, set_.tot_centroid):
            # Shuffle for every ov repetition
            if c % len(set_.ov_batch) == 0:
                if set_.ov_shuf == 1:
                    ov_shuf = set_.ov_batch.copy()
                    seedx_.shuffle(ov_shuf)
            # Set Uniform Distribution according to actual Centroid (loop through dstrb. 0-2)
            if c % len(set_.factorx) == 0:
                c_dstrb = 0
            else:
                c_dstrb += 1

            if set_.ov_type == 1:
                ov = float(ov_shuf[c % (len(ov_shuf))])
            else:
                ov = set_.ov_default
            # Create reference for 3x3 Grid
            gw = seedx_.randint(round(1 * set_.w_patch + set_.bordermargin), w - round(2 * 1 * set_.w_patch - set_.bordermargin))
            gh = seedy_.randint(round(1 * set_.h_patch + set_.bordermargin), h - round(2 * 1 * set_.h_patch - set_.bordermargin))

            for i in range(0, set_.no_patches):
                # Reference lies on first case
                if i == 0:
                    x1 = 0
                    y1 = 0
                else:
                    if set_.smpl_uniform == 1:
                        x1 = seedx_.randint(-set_.w_patch / set_.factorx[c_dstrb], +set_.w_patch / set_.factorx[c_dstrb])
                        y1 = seedy_.randint(-set_.h_patch / set_.factory[c_dstrb], +set_.h_patch / set_.factory[c_dstrb])
                    elif set_.smpl_uniform == 0:
                        if ov >= 0.99:  # Special case for 100% overlap
                            ovw = 0
                        else:
                            ovw = seedx_.randint(int((1 - ov) * -100), int((1 - ov) * 100)) / 100
                            x1 = int(ovw * set_.w_patch)
                            dw = set(list(range(gw, gw + set_.w_patch))).intersection(set(list(range(gw + x1, gw + x1 + set_.w_patch))))
                            dh = round((ov * set_.w_patch * set_.h_patch) / len(dw))
                            y1 = (set_.h_patch - dh) * (-1) ** seedx_.randint(2)

                # Create Data matrix with transformations of every image pair
                tx = x1
                ty = y1
                result = np.array([int(img_no), int(c), int(i), h, w, gw + x1, gw + x1 + set_.w_patch, gh + y1, gh + y1 + set_.h_patch, tx, ty, rot, scale, shear])
                pos_act = (img_no * set_.tot_centroid * set_.no_patches) + (c * set_.no_patches) + i
                patches[pos_act, :] = result

        # Output actual state
        if not img_no % 100:
            print('No: {0}/{1}'.format(img_no, len(files_)))
    print('No: {0}/{1}'.format(img_no + 1, len(files_)))
    return patches


def get_flow(dat_, set_, patch_r, patch_, lab_, lab_msk_):
    import cv2
    lab_ref = lab_.copy()
    lab_ref[lab_ref > 1] = 1
    # lab2d = np.zeros((set_.w_out, set_.h_out, 3), dtype=np.float32)
    lab2d = np.zeros((set_.w_patch, set_.h_patch, 3), dtype=np.float32)
    cols, rows = lab_msk_.shape

    # Create meshgrid (coordinate matrix)
    nx, ny = (rows, cols)
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv_flat = xv.flatten()
    yv_flat = yv.flatten()
    coords_ori = np.zeros((2, cols, rows), dtype=np.float32)
    coords_ori[0, :, :] = xv
    coords_ori[1, :, :] = yv
    coords = np.zeros((cols * rows, 2), dtype=np.float32)
    coords[:, 0] = xv_flat
    coords[:, 1] = yv_flat

    # Settings for registration
    if set_.reg_type == 'ref-mov':
        tx = -patch_[9]
        ty = -patch_[10]
        r = -patch_[11]
        s = -patch_[12]
        sh = -patch_[13]
    if set_.reg_type == 'mov-ref':
        tx = patch_[9]
        ty = patch_[10]
        r = patch_[11]
        s = patch_[12]
        sh = patch_[13]

    # Get transformation matrix
    M_t = None
    M_r = None
    M_s = None
    M_sh = None
    # Translation
    if tx != 0 or ty != 0:
        M_t = np.float64([[1, 0, tx], [0, 1, ty]])
    # Rotation
    if patch_[11] != 0:
        p_pivot = [patch_[5] + (0.5 * (patch_[6] - patch_[5])), patch_[7] + (0.5 * (patch_[8] - patch_[7]))]
        M_r = cv2.getRotationMatrix2D((p_pivot[0], p_pivot[1]), r, 1)
    # Scaling
    if patch_[12] != 0:
        s = -patch_[12]
        pts1 = np.float32([[patch_[5], patch_[7]],
                           [patch_[6], patch_[7]],
                           [patch_[6], patch_[8]]])
        pts2 = np.float32([[patch_[5] + s, patch_[7] + s],
                           [patch_[6] - s, patch_[7] + s],
                           [patch_[6] - s, patch_[8] - s]])
        M_s = cv2.getAffineTransform(pts2, pts1)
    # Shearing
    if patch_[13] != 0:
        sh = patch_[13]
        pts1 = np.float32([[patch_[5], patch_[7]],
                           [patch_[6], patch_[7]],
                           [patch_[6], patch_[8]]])
        pts2 = np.float32([[patch_[5] + sh, patch_[7]],
                           [patch_[6] + sh, patch_[7]],
                           [patch_[6], patch_[8]]])
        M_s = cv2.getAffineTransform(pts2, pts1)

    # Get coordinate mapping
    # Translation
    if M_t is not None:
        coords = cv2.transform(coords[None, :, :], M_t)
        coords = coords[0, :, :]
    # Rotation
    if M_r is not None:
        coords = cv2.transform(coords[None, :, :], M_r)
        coords = coords[0, :, :]
    # Scaling
    if M_s is not None:
        coords = cv2.transform(coords[None, :, :], M_s)
        coords = coords[0, :, :]
    # Shearing
    if M_sh is not None:
        coords = cv2.transform(coords[None, :, :], M_sh)
        coords = coords[0, :, :]

    # Reshape coordinate mapping, if no transformation copy zero coordinates back
    if M_t is not None or M_r is not None or M_s is not None or M_sh is not None:
        coords_inv = coords.T.reshape((-1, cols, rows))
        coords_res = coords_inv - coords_ori
    else:
        coords_res = np.zeros((2, cols, rows), dtype=np.float32)

    # Generate optical flow
    lab2d[:, :, 0] = coords_res[0, patch_r[7]: patch_r[8], patch_r[5]: patch_r[6]]
    lab2d[:, :, 1] = coords_res[1, patch_r[7]: patch_r[8], patch_r[5]: patch_r[6]]
    return lab2d


def get_labels(dat_, set_, files_, patches_):
    import cv2
    print('Write Labels...')
    no_files = len(patches_) - len(files_) * (set_.ov_batch.shape[0] - 1)
    data = []
    for patch_id in range(0, patches_.shape[0], set_.no_patches):
        patch_r = patches_[patch_id]
        lab_ref = np.zeros((patch_r[3], patch_r[4]), dtype=np.uint8)
        lab_ref[patch_r[7]: patch_r[8], patch_r[5]: patch_r[6]] = 1

        for idx in range(1, set_.no_patches):
            patch_m = patches_[patch_id + idx]

            # Settings for registration
            if set_.reg_type == 'ref-mov':
                tx = -patch_m[9]
                ty = -patch_m[10]
                r = -patch_m[11]
            if set_.reg_type == 'mov-ref':
                tx = -patch_m[9]
                ty = -patch_m[10]
                r = patch_m[11]

            # Affine Transformations
            # Translation
            lab_mov = np.zeros((patch_m[3], patch_m[4]), dtype=np.uint8)
            lab_mov[patch_m[7]: patch_m[8], patch_m[5]: patch_m[6]] = 1
            # Rotation
            if patch_m[11] != 0:
                px = patch_m[5] + (0.5 * (patch_m[6] - patch_m[5]))
                py = patch_m[7] + (0.5 * (patch_m[8] - patch_m[7]))
                M = cv2.getRotationMatrix2D((px, py), r, 1)
                lab_mov = cv2.warpAffine(lab_mov, M, (patch_m[4], patch_m[3]))
            # Scale
            if patch_m[12] != 0:
                lab = lab_ref + lab_mov
                lab[lab < 2] = 0
                lab[lab >= 2] = 1
                s = -patch_m[12]
                pts1 = np.float32([[patch_m[5], patch_m[7]],
                                   [patch_m[6], patch_m[7]],
                                   [patch_m[6], patch_m[8]]])
                pts2 = np.float32([[patch_m[5] + s, patch_m[7] + s],
                                   [patch_m[6] - s, patch_m[7] + s],
                                   [patch_m[6] - s, patch_m[8] - s]])
                M = cv2.getAffineTransform(pts1, pts2)
                lab_mov = cv2.warpAffine(lab, M, (patch_m[4], patch_m[3]))
            # Shear
            if patch_m[13] != 0:
                sh = patch_m[13]
                pts1 = np.float32([[patch_m[5], patch_m[7]],
                                   [patch_m[6], patch_m[7]],
                                   [patch_m[6], patch_m[8]]])
                pts2 = np.float32([[patch_m[5] + sh, patch_m[7]],
                                   [patch_m[6] + sh, patch_m[7]],
                                   [patch_m[6], patch_m[8]]])
                M = cv2.getAffineTransform(pts1, pts2)
                lab_mov = cv2.warpAffine(lab_mov, M, (patch_m[4], patch_m[3]))

            # Evaluate Intersection
            lab = lab_ref + lab_mov

            # Settings
            if set_.reg_type == 'ref-mov':
                lab = lab[patch_r[7]: patch_r[8], patch_r[5]: patch_r[6]]
            if set_.reg_type == 'mov-ref':
                # Correct rotation back
                if patch_m[11] != 0:
                    px = patch_m[5] + (0.5 * (patch_m[6] - patch_m[5]))
                    py = patch_m[7] + (0.5 * (patch_m[8] - patch_m[7]))
                    M = cv2.getRotationMatrix2D((px, py), -r, 1)
                    lab = cv2.warpAffine(lab, M, (patch_m[4], patch_m[3]))
                lab = lab[patch_m[7]: patch_m[8], patch_m[5]: patch_m[6]]

            # Scale Label output
            lab[lab < 2] = 0
            lab[lab >= 2] = 255

            # Calculate Overlap (round to 2 Digits)
            ov_int = round(get_ovarea(lab), 2)

            # Write Label image
            file_label = "doc" + str(patch_r[0]) + "_label_c" + str(patch_r[1]) + "_" + str(patch_r[2]) + "_" + str(idx) + ".jpg"
            cv2.imwrite(dat_.path_datas + file_label, lab)

            # Get optical flow
            if set_.flow_generated == 1:
                lab2d = get_flow(dat_, set_, patch_r, patch_m, lab, lab_ref)
                # Write flow image (-> .csv / .npy / .flo)
                file_flow = "doc" + str(patch_r[0]) + "_flow_c" + str(patch_r[1]) + "_" + str(patch_r[2]) + "_" + str(idx) + ".npy"
                np.save(dat_.path_datas + file_flow, lab2d)
            else:
                file_flow = ''

            # Write CSV Data
            f_ref = "doc" + str(patch_r[0]) + "_patch_c" + str(patch_r[1]) + "_" + str(patch_r[2]) + ".jpg"
            f_mov = "doc" + str(patch_m[0]) + "_patch_c" + str(patch_m[1]) + "_" + str(patch_m[2]) + ".jpg"
            temp = [[patch_id + idx, f_ref, f_mov, file_label, file_flow, ov_int, patch_m[9], patch_m[10], patch_m[11], patch_m[12], patch_m[13]]]
            data.extend(temp)

        # Output actual state
        if not patch_id % 1000:
            print('No: {0}/{1}'.format(patch_id, len(patches_) ), ' - Label:')
    print('No: {0}/{1}'.format(patch_id + 1, len(patches_) ), ' - Label:')
    return data


def wr_patchlist(dat_, set_, files_, patches_):
    import cv2
    print('Write Patches...')
    no_files = len(patches_) - len(files_) * (set_.ov_batch.shape[0] - 1)

    for patch_id, patch in enumerate(patches_):
        img, img_temp = [], []

        # Reference
        if patch_id == 0 or (patch_id % set_.no_patches) == 0:
            img = cv2.imread(dat_.path_datao + files_[int(patch[0])])
            img = get_rescale(img, set_, '')
            # Translation
            img_temp = img[patch[7]: patch[8], patch[5]: patch[6], :]
            # Save ref doc
            cv2.imwrite(dat_.path_datas + "doc" + str(patch[0]) + "_patch_c" + str(patch[1]) + "_" + str(patch[2]) + ".jpg",img_temp)

        # Moving
        else:
            img = cv2.imread(dat_.path_datao + files_[int(patch[0])])
            img = get_rescale(img, set_, '')

            # Affine Transformations
            if set_.reg_type == 'ref-mov':
                tx = patch[9]
                ty = patch[10]
                r = patch[11]
            if set_.reg_type == 'mov-ref':
                tx = -patch[9]
                ty = -patch[10]
                r = -patch[11]
            # Rotate
            if patch[11] != 0:
                px = patch[5] + (0.5 * (patch[6] - patch[5]))
                py = patch[7] + (0.5 * (patch[8] - patch[7]))
                M = cv2.getRotationMatrix2D((px, py), r, 1)
                img = cv2.warpAffine(img, M, (patch[4], patch[3]))
            # Scale
            if patch[12] != 0:
                s = -patch[12]
                pts1 = np.float32([[patch[5], patch[7]],
                                   [patch[6], patch[7]],
                                   [patch[6], patch[8]]])
                pts2 = np.float32([[patch[5] + s, patch[7] + s],
                                   [patch[6] - s, patch[7] + s],
                                   [patch[6] - s, patch[8] - s]])
                M = cv2.getAffineTransform(pts1, pts2)
                img = cv2.warpAffine(img, M, (patch[4], patch[3]))
            # Shear
            if patch[13] != 0:
                sh = patch[13]
                pts1 = np.float32([[patch[5], patch[7]],
                                   [patch[6], patch[7]],
                                   [patch[6], patch[8]]])
                pts2 = np.float32([[patch[5] + sh, patch[7]],
                                   [patch[6] + sh, patch[7]],
                                   [patch[6], patch[8]]])
                M = cv2.getAffineTransform(pts1, pts2)
                img = cv2.warpAffine(img, M, (patch[4], patch[3]))

            # Translation
            img_temp = img[patch[7]: patch[8], patch[5]: patch[6], :]
            # Save mov doc
            cv2.imwrite(dat_.path_datas + "doc" + str(patch[0]) + "_patch_c" + str(patch[1]) + "_" + str(patch[2]) + ".jpg", img_temp)

        # Output actual state
        if not patch_id % 1000:
            print('No: {0}/{1}'.format(patch_id, len(patches_)), ' - File:', dat_.path_datao + files_[int(patch[0])])
    print('No: {0}/{1}'.format(patch_id, len(patches_)), ' - File:', dat_.path_datao + files_[int(patch[0])])
    return 0


def get_affine(dat_, set_, patches_, seedx_):
    if set_.setup_rot == 1:
        no_val = len(patches_)
        rot_set = seedx_.randint(set_.rot_angle[0], set_.rot_angle[1], no_val)
        distribution = np.zeros(no_val, dtype=np.uint8)
        distribution[0:round(no_val * set_.rot_ratio)] = 1
        seedx_.shuffle(distribution)
        patches_[:, 11] = rot_set * distribution
    if set_.setup_scale == 1:
        no_val = len(patches_)
        scale_set = seedx_.randint(set_.scl_range[0], set_.scl_range[1], no_val)
        distribution = np.zeros(no_val, dtype=np.uint8)
        distribution[0:round(no_val * set_.scl_ratio)] = 1
        seedx_.shuffle(distribution)
        patches_[:, 12] = scale_set * distribution
    if set_.setup_shear == 1:
        no_val = len(patches_)
        scale_set = seedx_.randint(set_.shr_angle[0], set_.shr_angle[1], no_val)
        distribution = np.zeros(no_val, dtype=np.uint8)
        distribution[0:round(no_val * set_.shr_ratio)] = 1
        seedx_.shuffle(distribution)
        patches_[:, 13] = scale_set * distribution
    return patches_


def cr_data(dat, set_, set_type):
    start_time = time.time()
    # Get seeds
    if set_type == 'train':
        seedx = get_seed(set_.seed1)
        seedy = get_seed(set_.seed2)
    elif set_type == 'test':
        seedx = get_seed(set_.seed3)
        seedy = get_seed(set_.seed4)

    # Init
    init_folders(dat, set_)
    files = get_files(dat, set_, seedx)
    data_csv = [['No', 'Ref', 'Mov', 'Label', 'Flow', 'Overlap', 'Tx', 'Ty', 'Rot', 'Scale', 'Shear']]
    patch_csv = [['Img', 'Centroid', 'Patch', 'Height', 'Width', 'x1', 'x2', 'y1', 'y2', 'Tx', 'Ty', 'Rot', 'Scale', 'Shear']]

    # Get patches from filelist
    patch_list = get_patchlist(dat, set_, files, seedx, seedy)

    # Augment patches (rotation, scaling)
    patch_list2 = get_affine(dat, set_, patch_list, seedx)

    # Write patch-list and label-list
    wr_patchlist(dat, set_, files, patch_list2)
    label_csv = get_labels(dat, set_, files, patch_list)
    data_csv.extend(label_csv)
    patch_csv.extend(patch_list2)

    # Write CSV files
    if set_.cr_csv == 1:
        f = open(dat.path_dataraw + dat.path_csv_label, 'w')  # w = overwrite / a = append
        a = csv.writer(f)
        a.writerows(data_csv)
        f.close()
        f = open(dat.path_dataraw + dat.path_patchlist, 'w')  # w = overwrite / a = append
        a = csv.writer(f)
        a.writerows(patch_csv)
        f.close()

    # Output states
    stop_time = time.time()
    print(set_type, '-' * 30)
    print('Time / File:', (stop_time - start_time) / len(patch_list))
    print('Time needed:', stop_time - start_time)
    print('Tot. pairs :', patch_list.shape[0] - 1)
    print('-' * 30)


def main():
    # Get initialization
    dat_train = get_train()
    dat_test = get_test()
    setting = get_set()

    if setting.cr_train == 1:
        cr_data(dat_train, setting, 'train')
    if setting.cr_test == 1:
        cr_data(dat_test, setting, 'test')


if __name__ == '__main__':
    main()
