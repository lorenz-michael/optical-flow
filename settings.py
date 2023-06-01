class Setting(object):
    pass


''' TrainData Settings'''
def get_train():
    dat = Setting()
    # Path settings
    dat.path_dataraw = 'data/'
    dat.path_code = dat.path_dataraw + '/code/3_FlowNet/'
    dat.path_datao = dat.path_dataraw + 'train_raw/'
    dat.path_datas = dat.path_dataraw + 'train/'
    dat.path_info = dat.path_dataraw + 'train/info/'
    dat.path_pred = dat.path_dataraw + 'pred/'
    # Filenames
    dat.path_csv_label = 'labels_train.csv'
    dat.path_hdf5 = 'train.hdf5'
    dat.path_hdf5_scale = 'train_p.hdf5'
    dat.path_patchlist = 'patchlist_train.csv'
    dat.mdl_name_unet1 = 'mdl_unet.hdf5'
    dat.mdl_name_siam1 = 'mdl_siam.hdf5'
    dat.path_model_flow1 = 'mdl_flow.hdf5'
    return dat


''' TestData Settings'''
def get_test():
    dat = Setting()
    # Path setting
    dat.path_dataraw = 'data/'
    dat.path_code = 'data/3_FlowNet/'
    dat.path_pred = 'data/3_FlowNet/pred/'
    dat.path_datao = dat.path_dataraw + 'test_raw/'
    dat.path_datas = dat.path_dataraw + 'test/'
    dat.path_info = dat.path_dataraw + 'test/info/'
    dat.path_pred = dat.path_dataraw + 'pred/'
    dat.path_score1 = dat.path_dataraw + 'scores/cnn/'
    dat.path_score2 = dat.path_dataraw + 'scores/sift/'
    # Filenames
    dat.path_csv_label = 'labels_test.csv'
    dat.path_hdf5 = 'test.hdf5'
    dat.path_hdf5_scale = 'test_p.hdf5'
    dat.path_patchlist = 'patchlist_test.csv'
    dat.file_score1 = '0_scores_cnn.csv'
    dat.file_score2 = '0_scores_sift.csv'
    dat.file_score3 = 'scores_cnn'
    dat.file_score4 = 'scores_sift'
    dat.mdl_name_unet1 = 'mdl_unet.hdf5'
    dat.mdl_name_siam1 = 'mdl_siam.hdf5'
    dat.path_model_flow1 = 'mdl_flow.hdf5'
    return dat


''' General Settings'''
def get_set():
    import numpy as np
    setting = Setting()

    # RGB setting
    setting.rgb_data = 1            # define if RGB support

    # General setting
    setting.cr_train = 1
    setting.cr_test = 1
    setting.cr_hdf5 = 1
    setting.cr_hdf5_scale = 1
    setting.blk_size = 5000         # no files per hdf5 batch writing (memory depend.)
    setting.cr_label = 1
    setting.cr_csv = 1
    setting.cr_mark = 1
    setting.cr_orig = 1
    setting.hdf5compr = 1
    setting.hdf5comprlevel = 2      # from 1 to 9 highest

    # Training setting
    setting.diag = 0
    setting.re_train = 0            # select if pre-trained weights are loaded
    setting.loss = 'ed_norm_sum2'   # dice, binary_xe, xe, mse, ed_clip, l1_loss, l2_loss, ed_ssd_loss
    setting.mdl_type = 'flownet4'   # unet, siamese, flownet
    setting.batch_shuffle = 0       # batch shuffling before each epoch
    setting.no_epochs = 10         # epochs
    setting.b_size = 4              # batch size
    setting.learning_rate = 1e-5    # learning rate
    setting.p_valid = 0.1           # size of validation set
    setting.tot_set = 2             # no of training sets per epoch

    # Data shuffle
    setting.datashuffle = 1         # shuffle images from folder
    setting.shuffle_train = 0       # shuffle images before storing to HDF5 file (only TRAIN data)
    setting.shuffle_test = 0        # shuffle images before storing to HDF5 file (only TRAIN data)

    # Input setting
    setting.ch_img = 6              # number of input channel (2 RGB = 6 channels)
    setting.h_rescale = 1           # factor for rescaling images in height
    setting.w_rescale = 1           # factor for rescaling images in width
    setting.data_normalize = 1      # remove MEAN/STD from input data
    setting.data_center = 0         # data rescaling for images, 0=[0, 255], 1=[0, 1], 2=[-0.5,+0.5]

    # Uniform settings
    setting.smpl_uniform = 0        # select uniform sampling
    setting.bordermargin = 0        # select size of bordermarging
    setting.factorx = [2, 1.4, 1]   # distribution correction for moving coord x (1=difficult, 1.4=middle, 2=easier)
    setting.factory = [2, 1.4, 1]   # distribution correction for moving coord y

    # Non-uniform settings (Translations)
    setting.ov_type = 1             # 0=default, 1=steps
    setting.ov_shuf = 1
    setting.ov_default = 0.5        # default overlap
    setting.ov_step = 0.05          # 0.05
    setting.ov_start = 0.5          # 0.001
    setting.ov_stop = 0.95
    setting.ov_batch = np.arange(setting.ov_start, setting.ov_stop + setting.ov_step, setting.ov_step)

    # CNN settings
    setting.h_out = 128             # cnn input in pixel
    setting.w_out = 128             # cnn input in pixel
    setting.initializer = 'glorot_uniform'  # XAVIER = glorot_uniform, glorot_normal, he_uniform ,he_normal

    # Patches / Centroid setting
    setting.h_patch = 128           # patch output in pixel
    setting.w_patch = 128           # patch output in pixel
    setting.no_patches = 2          # no of patches per centroids
    setting.no_centroid = 1         # no of centroids
    setting.tot_centroid = setting.no_centroid * len(setting.ov_batch)

    # Settings registration
    # setting.reg_type = 'ref-mov'
    setting.reg_type = 'mov-ref'    # ref-mov, mov-ref, define registration way

    # Setting optical flow
    setting.flow_generated = 1      # define if flow is generated
    setting.flow_full = 1           # define if flow generated only in overlapping area

    # General SIFT settings
    setting.sift_rot = 1            # define SIFT model (0=2pt translation, 1=3pt rotation)
    setting.clahe = 1               # define if CLAHE pre-processing is applied (contrast enhancement)
    setting.plot_keypoints = 0

    # Data Augmentation (Noise, Blurr, Contrast, Gamma, Channel shift)
    setting.dataugm = 1
    setting.dataugm_ratio = 0.5         # amount
    setting.dataugm_level = 10          # number of augmentation levels
    setting.noise_gauss = [0, 0.2]      # 0, 0.5
    setting.noise_sp = [0, 0.2]         # 0, 0.5
    setting.blur_gauss = [0, 2]         # 0,3
    setting.blur_median = [1, 3]        # 1,5
    setting.light_contrast = []         # statically defined in data_augment.py
    setting.light_gamma = [0.6, 1.4]
    setting.ch_shift_range = [0, 3]     # select channels to shift

    # Data Augmentation (Rotation)
    setting.setup_rot = 1           # select rotation augmentation
    setting.rot_ratio = 0.5         # amount
    setting.rot_angle = [-30, +30]  # range in degree

    # Data Augmentation (Scale)
    setting.setup_scale = 0         # select scaling augmentation
    setting.scl_ratio = 0.5         # amount
    setting.scl_range = [-40, +40]  # range in pixels

    # Data Augmentation (Shear)
    setting.setup_shear = 0         # select shearing augmentation
    setting.shr_ratio = 1        # amount
    setting.shr_angle = [-40, +40]  # range in pixels

    # Data Augmentation (Flow)
    setting.flow_scaled = 1         # pixel=[-pixel, +pixel], pixel_pos=[0, +2*pixel], 0=[-1,+1], 1=[0,1],  2=[0,2], 8=[0,8]
    setting.flow_penalize = 0       # define if non flow zones are filled with high values to penalize Loss
    setting.flow_penalize_value = 0

    # Evaluation settings
    setting.p_tresh = 0.6

    # Setting randomization
    setting.seed1 = 123
    setting.seed2 = 321
    setting.seed3 = 456
    setting.seed4 = 654
    return setting


''' Setting for SIFT weak'''
def get_evalset():
    evalset = Setting()
    evalset.MIN_MATCH_COUNT = 4
    evalset.nfeatures = 0
    evalset.nOctaveLayers = 3
    evalset.contrastThreshold = 0.02
    evalset.edgeThreshold = 10
    evalset.sigma = 1.6
    return evalset


'''Setting for SIFT strong (detect more keypoints, slower)'''
def get_evalset2():
    evalset = Setting()
    evalset.MIN_MATCH_COUNT = 4
    evalset.nfeatures = 0
    evalset.nOctaveLayers = 3
    evalset.contrastThreshold = 0.005  # 0.005
    evalset.edgeThreshold = 15  # 15
    evalset.sigma = 1.6
    return evalset
