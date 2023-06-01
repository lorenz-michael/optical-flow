

import csv
import numpy as np
import time
import tensorflow as tf

from keras import backend as K
from keras.callbacks import Callback as KCallback
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model

from utils.dataset import get_seed, get_set, get_train
from utils.data_augment import get_augmentationlist, get_augmentation_batch
from utils.convert_hdf5 import check_hdf5, rd_hdf5_batch, preprocess_scipy

from utils.models import get_unet
from utils.models import get_flownet, get_flownet2, get_flownet4

# from data import load_train_data, load_test_data
#K.set_image_data_format('channels_first')       # Theano dimension ordering
K.set_image_data_format('channels_last')       # Theano dimension ordering


# Constants
PI = 3.1415926535


class MyCallback(KCallback):
    '''
    alpha = K.variable(0.5)
    beta = K.variable(0.5)
    model.compile(..., loss_weights=[alpha, beta], ...)
    '''

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        self.alpha = self.alpha - 0.1
        self.beta = self.beta + 0.1


class LossHistory(KCallback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.losses = []
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        csv_append('history_loss.csv', logs, epoch, self.times[0])
        return


class LossHistory_multi(KCallback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.losses = []
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return

    def on_epoch_end(self, epoch, logs={}):
        csv_append_multi('history_loss.csv', logs, epoch, 0)
        return


class LossHistory_reg(KCallback):
    def on_train_begin(self, logs={}):
        self.losses = []
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return

    def on_epoch_end(self, epoch, logs={}):
        csv_append_reg('history_loss.csv', logs, epoch, 0)
        return


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=1)


def mean_squared_error_reg(y_true, y_pred):
    # Correct init shape of y_pred (correct empty tensor)
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value + 3, sh[2]._value, sh[3]._value))
    # Split reference into flow, intensities and Transformation Matrix
    y_true_flow = y_true[:, :, :, 0:2]
    y_pred_flow = y_pred[:, :, :, 0:2]
    y_pred_flow = K.clip(y_pred, -1e7, 1e7)
    mean_squared_error_reg = K.mean(K.square(y_pred_flow - y_true_flow), axis=3)
    return mean_squared_error_reg


def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=3, keepdims=True))


def l2_norm_sum(y_true, y_pred):
    return K.sum(K.sum(K.sum(K.square(y_true - y_pred), axis=3, keepdims=True), axis=1, keepdims=True), axis=2, keepdims=True)


def ed_clip(y_true, y_pred):
    y_pred = K.clip(y_pred, -1e7, 1e7)
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=3, keepdims=True))


def ed_sum2(y_true, y_pred):
    y_pred = K.clip(y_pred, -1e7, 1e7)
    return K.sum(K.sum(K.sqrt(K.sum(K.square(y_true - y_pred), axis=3, keepdims=True)), axis=1, keepdims=True), axis=2, keepdims=True)


def ed_norm_sum2(y_true, y_pred):
    # # Settings
    # alpha = 2.0  # 1.5
    # beta = -2.0  # -1.5
    # # Correct init shape of y_pred (correct empty tensor)
    # sh = y_pred.get_shape()
    # #y_true.set_shape((sh[0]._value, sh[1]._value, sh[2]._value, sh[3]._value))  #Modified
    # y_true.set_shape((sh[0], sh[1], sh[2], sh[3]))
    # y_pred = K.clip(y_pred, -1e7, 1e7)
    # y_true_flow = y_true[:, 0:2, :, :]
    #
    # ed = K.sqrt(K.sum(K.square(y_true_flow - y_pred), axis=1, keepdims=True))
    # ed_mag = K.sqrt(K.sum(K.square(y_true_flow), axis=1, keepdims=True))
    # m_weight = K.exp(ed * alpha) * K.exp(ed_mag * beta)
    # loss = ed * m_weight
    # loss = K.sum(K.sum(loss, axis=2, keepdims=True), axis=3, keepdims=True)

    # Settings
    alpha = 2.0  # 1.5
    beta = -2.0  # -1.5
    # Correct init shape of y_pred (correct empty tensor)
    sh = y_pred.get_shape()
    # y_true.set_shape((sh[0]._value, sh[1]._value, sh[2]._value, sh[3]._value))  #Modified
    y_true.set_shape((sh[0], sh[1], sh[2], sh[3]))
    y_pred = K.clip(y_pred, -1e7, 1e7)
    y_true_flow = y_true[:, :, :, 0:2]

    ed = K.sqrt(K.sum(K.square(y_true_flow - y_pred), axis=3, keepdims=True))
    ed_mag = K.sqrt(K.sum(K.square(y_true_flow), axis=3, keepdims=True))
    m_weight = K.exp(ed * alpha) * K.exp(ed_mag * beta)
    loss = ed * m_weight
    loss = K.sum(K.sum(loss, axis=1, keepdims=True), axis=2, keepdims=True)
    return loss


def ed_smooth(y_true, y_pred):
    e = 0.01
    alpha = 0.1
    # Clip predicted flow
    y_pred = K.clip(y_pred, -1e7, 1e7)
    # Get Tx and Ty
    y_predx = y_pred[:, :, :, 0]
    y_predy = y_pred[:, :, :, 1]
    # Get s (correction of -0.5, due to prior flow scaling)
    s = K.square(y_predx-0.5) + K.square(y_predy-0.5)
    # Calculate PSI
    psi = K.sqrt(K.square(s) + K.square(e))
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=3, keepdims=True)) + (alpha * psi)


def ed_smooth_sum(y_true, y_pred):
    e = 0.01
    alpha = 0.1
    # Correct init shape of y_pred (correct empty tensor)
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value + 3, sh[2]._value, sh[3]._value))
    y_true_flow = y_true[:, :, :, 0:2]
    y_pred_flow = y_pred[:, :, :, 0:2]
    # Clip predicted flow
    y_pred_flow = K.clip(y_pred_flow, -1e7, 1e7)
    # Get Tx and Ty
    y_predx = y_pred[:, :, :, 0]
    y_predy = y_pred[:, :, :, 1]
    y_predx = K.expand_dims(y_predx, axis=3)
    y_predy = K.expand_dims(y_predy, axis=3)
    # Get s (correction of -0.5, due to prior flow scaling)
    s = K.square(y_predx-0.5) + K.square(y_predy-0.5)
    # Calculate PSI
    psi = K.sqrt(K.square(s) + K.square(e))
    ed = K.sqrt(K.sum(K.square(y_true_flow - y_pred_flow), axis=3, keepdims=True))
    loss = ed + (alpha * psi)
    loss = K.sum(K.sum(loss, axis=1, keepdims=True), axis=2, keepdims=True)
    return loss


def ed_ae_sum(y_true, y_pred):
    alpha_ed = 1
    alpha_ae = 0.3
    # Correct init shape of y_pred (correct empty tensor)
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value + 3, sh[2]._value, sh[3]._value))
    y_true_flow = y_true[:, :, :, 0:2]
    y_pred_flow = y_pred[:, :, :, 0:2]
    # Clip predicted flow
    y_pred_flow = K.clip(y_pred, -1e7, 1e7)
    # Calculate ED + AE
    ed = K.sqrt(K.sum(K.square(y_true_flow - y_pred_flow), axis=31, keepdims=True))
    ae = angular_error(y_true_flow, y_pred_flow)
    ae = K.expand_dims(ae, axis=1)
    # loss = (alpha_ed * ed) + (alpha_ae * ae)
    # Sum ED + Sum AE
    ed_sum = K.sum(K.sum(ed, axis=1, keepdims=True), axis=2, keepdims=True)
    ae_sum = K.sum(K.sum(ae, axis=1, keepdims=True), axis=2, keepdims=True)
    loss = (alpha_ed * ed_sum) + (alpha_ae * ae_sum)

    loss = K.clip(loss, 0, 1e7)
    return loss


def angular_error(y_true, y_pred):
    # Settings
    epsilon = 1e-6
    # Rescale flow from [0, 1] to [-1, +1] range
    y_true -= 0.5
    y_true *= 2
    y_pred -= 0.5
    y_pred *= 2
    uest = y_pred[:, :, :, 0]
    vest = y_pred[:, :, :, 1]
    uref = y_true[:, :, :, 0]
    vref = y_true[:, :, :, 1]
    # 3D Vector Angle
    nominator = (uest * uref) + (vest * vref) + 1 - epsilon
    denominator = K.sqrt(K.square(uest) + K.square(vest) + 1) * K.sqrt(K.square(uref) + K.square(vref) + 1 - epsilon)
    ae = tf.acos((nominator / denominator))
    # Clip ae
    ae = K.clip(ae, 0, 3.141)
    return ae


def ed_border_sum(y_true, y_pred):
    # Define init shape of y_pred (correct empty tensor)
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value + 3, sh[2]._value, sh[3]._value))
    # Reshape Tensors
    y_predx = y_pred[:, :, :, 0]
    y_predy = y_pred[:, :, :, 1]
    y_truex = y_true[:, :, :, 0]
    y_truey = y_true[:, :, :, 1]
    y_predx = K.expand_dims(y_predx, axis=3)
    y_predy = K.expand_dims(y_predy, axis=3)
    y_truex = K.expand_dims(y_truex, axis=3)
    y_truey = K.expand_dims(y_truey, axis=3)

    # Average the predicted flow, to compensate zeropadding
    # Create binary mask
    border = 1
    msk_in = np.zeros((128, 128), dtype=np.int)
    msk_in[border:128 - border, border:128 - border] = 1
    msk_out = np.ones((128, 128), dtype=np.int)
    msk_out -= msk_in
    K_msk_in = K.variable(value=msk_in)
    K_msk_out = K.variable(value=msk_out)

    # Average Pool
    uest = K.pool2d(y_predx, (3, 3), strides=(1, 1), padding='same', pool_mode='avg')
    vest = K.pool2d(y_predy, (3, 3), strides=(1, 1), padding='same', pool_mode='avg')

    # Apply binary mask
    uest *= K_msk_out
    vest *= K_msk_out
    y_predx *= K_msk_in
    y_predy *= K_msk_in
    y_predx += uest
    y_predy += vest

    ed = K.sum(K.sqrt(K.square(y_truex - y_predx) + K.square(y_truey - y_predy)), axis=3, keepdims=True)
    ed_sum = K.sum(K.sum(ed, axis=1, keepdims=True), axis=2, keepdims=True)
    loss = ed_sum
    return ed


def l1_loss(y_true, y_pred):
    # Define init shape of y_pred (correct empty tensor)
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value + 3, sh[2]._value, sh[3]._value))
    # Reshape Tensors
    uest = y_pred[:, :, :, 0]
    vest = y_pred[:, :, :, 1]
    uref = y_true[:, :, :, 0]
    vref = y_true[:, :, :, 1]
    uest = K.expand_dims(uest, axis=3)
    vest = K.expand_dims(vest, axis=3)
    uref = K.expand_dims(uref, axis=3)
    vref = K.expand_dims(vref, axis=3)
    # Get L1 Loss
    l1_err_u = K.sum(K.abs(uref - uest), axis=3, keepdims=True)
    l1_err_v = K.sum(K.abs(vref - vest), axis=3, keepdims=True)
    loss = l1_err_u + l1_err_v
    loss = K.clip(loss, -1e7, 1e7)
    return loss


def l2_loss(y_true, y_pred):
    # Define init shape of y_pred (correct empty tensor)
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value + 3, sh[2]._value, sh[3]._value))
    # Reshape Tensors
    uest = y_pred[:, :, :, 0]
    vest = y_pred[:, :, :, 1]
    uref = y_true[:, :, :, 0]
    vref = y_true[:, :, :, 1]
    uest = K.expand_dims(uest, axis=3)
    vest = K.expand_dims(vest, axis=3)
    uref = K.expand_dims(uref, axis=3)
    vref = K.expand_dims(vref, axis=3)

    # Get l2 Loss
    l2_err_u = K.sum(K.square(uref - uest), axis=3, keepdims=True)
    l2_err_v = K.sum(K.square(vref - vest), axis=3, keepdims=True)

    #  Create binary mask for Loss (remove border pixels from keras zero padding)
    border = 5
    msk_in = np.zeros((128, 128), dtype=np.int)
    msk_in[border:128-border, border:128-border] = 1
    msk_out = np.ones((128, 128), dtype=np.int)
    msk_out -= msk_in
    K_msk_in = K.variable(value=msk_in)
    K_msk_out = K.variable(value=msk_out)

    # Get Tensor mean value
    u_pool = K.pool2d(l2_err_u, (8, 8), strides=(1, 1), padding='same', pool_mode='avg')  #8,8
    u_pool *= K_msk_out
    v_pool = K.pool2d(l2_err_v, (8, 8), strides=(1, 1), padding='same', pool_mode='avg')
    v_pool *= K_msk_out

    # Apply binary mask
    l2_err_u *= K_msk_in
    l2_err_u += u_pool
    l2_err_v *= K_msk_in
    l2_err_v += v_pool
    loss = l2_err_u + l2_err_v
    loss = K.clip(loss, -1e7, 1e7)
    return loss


def loc_smooth_loss(y_true, y_pred):
    # Settings
    alpha_ed = 1
    alpha_sm = 0.1  # 0.001
    # Correct init shape of y_pred (correct empty tensor)
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value + 3, sh[2]._value, sh[3]._value))
    # Split reference into flow, I1 and I2
    y_true_flow = y_true[:, :, :, 0:2]
    y_pred_flow = y_pred[:, :, :, 0:2]
    I1 = y_true[:, :, :, 2]
    I2 = y_true[:, :, :, 3]
    label = y_true[:, :, :, 4]

    # Calculate ED + local smoothness
    l2 = K.sum(K.square(y_true_flow - y_pred_flow), axis=3, keepdims=True)
    sm = K.pool2d(y_pred_flow, (3, 3), strides=(1, 1), padding='same', pool_mode='avg')
    sm_diff = K.sum(K.abs(y_pred_flow - sm), axis=3, keepdims=True)
    loss = (alpha_ed * l2) + (alpha_sm * sm_diff)
    loss = K.clip(loss, 0, 1e7)
    return loss


def laplacian_smooth(y_true, y_pred):
    # Reshape Tensors
    y_predx = y_pred[:, :, :, 0]
    y_predy = y_pred[:, :, :, 1]
    y_truex = y_true[:, :, :, 0]
    y_truey = y_true[:, :, :, 1]
    y_predx = K.expand_dims(y_predx, axis=3)
    y_predy = K.expand_dims(y_predy, axis=3)
    y_truex = K.expand_dims(y_truex, axis=3)
    y_truey = K.expand_dims(y_truey, axis=3)

    y_x = y_truex - y_predx
    y_y = y_truey - y_predy

    kernel = [[0, 0, 1, 0, 0],
              [0, 1, 2, 1, 0],
              [1, 2, -16, 2, 1],
              [0, 1, 2, 1, 0],
              [0, 0, 1, 0, 0]]
    K_kernel = K.variable(value=kernel)

    y_x = K.conv2d(y_x, K_kernel, strides=(1, 1), padding='valid', data_format="channels_first")
    y_y = K.conv2d(y_y, K_kernel, strides=(1, 1), padding='valid', data_format="channels_first")
    loss = y_x + y_y
    return loss


def ed_irl_loss(y_true, y_pred):
    # Clip predictions
    y_pred = K.clip(y_pred, -1e7, 1e7)
    # Correct init shape of y_pred (correct empty tensor)
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value + 3, sh[2]._value, sh[3]._value))
    # Split reference into flow, I1 and I2
    y_true_flow = y_true[:, :, :, 0:2]
    flow_pred = y_pred[:, :, :, 0:2]
    I1 = y_true[:, :, :, 2]
    I2 = y_true[:, :, :, 3]
    label = y_true[:, :, :, 4]
    # Expand dimensions
    I1 = K.expand_dims(I1, axis=3)
    I2 = K.expand_dims(I2, axis=3)
    label = K.expand_dims(label, axis=3)

    # Scale Flow and Images
    flow_pred -= 0.5
    flow_pred *= (2*128)
    K.round(flow_pred)
    I1 *= 255
    I2 *= 255

    # Create meshgrid (reference coordinate matrix)
    rows = 128
    cols = 128
    nx, ny = (rows, cols)
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    xu, yv = np.meshgrid(x, y)
    K_xu = K.variable(value=xu)
    K_yv = K.variable(value=yv)

    # Get coordinates for warped image (predicted flow)
    K_xu = K.round(K_xu)
    K_yv = K.round(K_yv)
    K_u = K.round(flow_pred[:, :, :, 0])
    K_v = K.round(flow_pred[:, :, :, 1])
    K_xu_warp = K.clip(K_xu + K_u, -1, rows)
    K_yv_warp = K.clip(K_yv + K_v, -1, cols)
    K_xu_warp = tf.to_int32(K_xu_warp)
    K_yv_warp = tf.to_int32(K_yv_warp)

    # Get coordinates for reference label
    K_u_r = K.round(y_true_flow[:, 0, :, :])
    K_v_r = K.round(y_true_flow[:, 1, :, :])
    K_xu_warp_r = K.clip(K_xu + K_u_r, -1, rows)
    K_yv_warp_r = K.clip(K_yv + K_v_r, -1, cols)
    K_xu_warp_r = tf.to_int32(K_xu_warp_r)
    K_yv_warp_r = tf.to_int32(K_yv_warp_r)

    # Get warped image I2
    K_I2 = K.permute_dimensions(I2, (0, 2, 3, 1))
    K_I2 = get_pixel_value(K_I2, K_xu_warp, K_yv_warp)
    K_I2 = K.permute_dimensions(K_I2, (0, 3, 1, 2))

    # Get warped Label
    K_label = K.permute_dimensions(label, (0, 2, 3, 1))
    K_label = get_pixel_value(K_label, K_xu_warp_r, K_yv_warp_r)
    K_label = K.permute_dimensions(K_label, (0, 3, 1, 2))

    # Get intensity differences and rescale intensities
    K_diff = K.abs(I1 - K_I2)
    K_diff /= 255

    # Mask overlapping area
    loss_IRL = K_diff * K_label
    loss_IRL = K.sum(K.sum(K.sum(loss_IRL, axis=1, keepdims=True), axis=2, keepdims=True), axis=3, keepdims=True)

    # Calculate Loss (l2 + IRL)
    l2 = K.sum(K.square(y_true[:, 0:2, :, :] - y_pred[:, 0:2, :, :]), axis=1, keepdims=True)
    l2_sum = K.sum(K.sum(l2, axis=2, keepdims=True), axis=3, keepdims=True)
    loss = l2_sum + (0.1 * loss_IRL)
    loss = K.clip(loss, 0, 1e7)
    return loss


def get_pixel_value(img, x, y):
    # We assume that x and y have the same shape.
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    # Create a tensor that indexes into the same batch. (for gather_nd)
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))
    indices = tf.stack([b, y, x], 3)
    img_t = tf.gather_nd(img, indices)
    return img_t


def l1_smooth_loss(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = tf.where(tf.less(x, 1.0), 0.5 * x ** 2, x - 0.5)
    return K.sum(x, axis=1)


def manhattan_distance(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred), axis=1, keepdims=True)


def siamese_euclidean(y_true, y_pred):
    y_predx = K.eval(y_pred[:, :, :, 0])
    y_predy = K.eval(y_pred[:, :, :, 1])
    y_truex = K.eval(y_true[:, :, :, 0])
    y_truey = K.eval(y_true[:, :, :, 1])
    res = np.sqrt(np.add(np.square(y_truex - y_predx), np.square(y_truey - y_predy)))
    return res


def siamese_euclidean_t(y_true, y_pred):
    y_predx = y_pred[:, :, :, 0]
    y_predy = y_pred[:, :, :, 1]
    y_truex = y_true[:, :, :, 0]
    y_truey = y_true[:, :, :, 1]
    res = K.sqrt(K.square(y_truex - y_predx) + K.square(y_truey - y_predy))
    return res


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def coeff_determination(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - ss_res / (ss_tot + K.epsilon()))


def csv_init(set_, file):
    if set_.loss == 'ed_multi':
        temp = [['Epochs', 'Train Weighted Loss', 'Output Loss', 'Loss Flow9', 'Loss Flow8', 'Loss Flow7', 'Loss Flow6',
                 'Val weighted Loss', 'Val Output Loss', 'Val Loss Flow9', 'Val Loss Flow8', 'Val Loss Flow7',
                 'Val Loss Flow6', 'Time']]
    elif set_.loss in ('ed_ae_sum', 'l2_loss', 'loc_smooth_loss'):
        temp = [['Epochs', 'Train Loss', 'Train MSE', 'Val Loss', 'Val MSE', 'Time']]
    else:
        temp = [['Epochs', 'Loss Train', 'Dice Train', 'MSE Train', 'ACC Train',
                 'Loss Validation', 'Dice Validation', 'MSE Validation', 'ACC Validation', 'Time']]
    csv_write(file, temp, 'w')


def csv_write(file, data, arg):
    b = open(file, arg)
    a = csv.writer(b)
    a.writerows(data)
    b.close()


def csv_append(file, data, epochs, time_eta):
    temp = ([[epochs, data['loss'], data['dice_coef'], data['mse'], data['accuracy'],
              data['val_loss'], data['val_dice_coef'], data['val_mse'], data['val_accuracy'], time_eta]])
    csv_write(file, temp, 'a')


def csv_append_multi(file, data, epochs, time_eta):
    temp = ([[epochs, data['loss'], data['main_output_loss'], data['flow9_up_loss'], data['flow8_up_loss'], data['flow7_up_loss'], data['flow6_up_loss'],
              data['val_loss'], data['val_main_output_loss'], data['val_flow9_up_loss'], data['val_flow8_up_loss'], data['val_flow7_up_loss'], data['val_flow6_up_loss'],
             time_eta]])
    csv_write(file, temp, 'a')


def csv_append_reg(file, data, epochs, time_eta):
    temp = ([[epochs, data['loss'], data['mean_squared_error_reg'], data['val_loss'], data['val_mean_squared_error_reg'], time_eta]])
    csv_write(file, temp, 'a')


def show_setting(set_):
    print('-' * 30)
    print('Settings:')
    print('Height:', set_.h_out, ' / Width:', set_.w_out)
    print('Batch Size:', set_.b_size, ' / Epochs:', set_.no_epochs)
    print('-' * 30)
    print('Learning-Rate:', set_.learning_rate)
    print('-' * 30)


def show_setting_reduced(set_):
    print('-' * 30)
    print('Model:        ', set_.mdl_type)
    print('-' * 30)
    print('Loss:         ', set_.loss)
    print('-' * 30)
    print('Height:', set_.h_out, ' / Width:', set_.w_out)
    print('Batch Size:   ', set_.b_size)
    print('Epochs:       ', set_.no_epochs)
    print('Learning-Rate:', set_.learning_rate)
    print('Load weights: ', set_.re_train)
    print('-' * 30)
    print('Data normalize', set_.data_normalize)
    print('Batch shuffle:', set_.batch_shuffle)
    print('Weight init:  ', set_.initializer)
    if set_.dataugm == 0: print('Data augm.:    NO')
    if set_.dataugm == 1: print('Data augm.:    YES - ', set_.dataugm_ratio * 100, '%')
    if set_.data_center == 0: print('Data scale:    0, 255')
    if set_.data_center == 1: print('Data scale:    0, 1')
    if set_.data_center == 2: print('Data scale:    -0.5, +0.5')
    if set_.flow_scaled == 'pixel': print('Flow scale:    pixels')
    if set_.flow_scaled == 0: print('Flow scale:    -1, 1')
    if set_.flow_scaled == 1: print('Flow scale:    0, 1')
    if set_.flow_scaled == 2: print('Flow scale:    0, 2')


def fit_train_batch(dat, set_):
    print('-' * 30)
    print('Load preprocessed train data...')
    print('-' * 30)

    # Get Seed values
    seedx = get_seed(set_.seed1)

    # Calculate Set Elements
    tot_files = check_hdf5(dat.path_dataraw + dat.path_hdf5_scale, set_)        #Path modified
    tot_valid = int(tot_files * set_.p_valid)
    tot_test = tot_files - tot_valid
    s_size = int(tot_test / set_.tot_set)

    start_valid = tot_test
    stop_valid = tot_files
    set_test = np.arange(0, tot_files - tot_valid, s_size)

    # Create Model
    if set_.mdl_type == 'unet':
        model = get_unet(set_)
        weight_name = dat.mdl_name_unet1
    if set_.mdl_type == 'flownet1':
        model = get_flownet(set_)
        weight_name = dat.path_model_flow1
    if set_.mdl_type == 'flownet2':
        model = get_flownet2(set_)
        weight_name = dat.path_model_flow1
    if set_.mdl_type == 'flownet4':
        model = get_flownet4(set_)
        weight_name = dat.path_model_flow1

    # Load previous calculated weights, if selected
    if set_.re_train == 1:
        placeholder = []
        if set_.mdl_type in 'unet':
            model.load_weights('model111_ep12.hdf5')
        if set_.mdl_type == 'siamese':
            model.load_weights('mdl_siam.hdf5')
        if set_.mdl_type in ('flownet1', 'flownet2','flownet4'):
            model.load_weights('mdl_flow.hdf5')
    # Compile model with selected Loss
    # L1 and l2 losses
    if set_.loss == 'l1_loss':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[l1_loss],
                      metrics=[mean_squared_error_reg])
    if set_.loss == 'l2_loss':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[l2_loss],
                      metrics=[mean_squared_error_reg])
    if set_.loss == 'l2_norm_sum':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[l2_norm_sum],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])

    # Intensity and local smoothness losses
    if set_.loss == 'ed_irl_loss':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[ed_irl_loss],
                      metrics=[mean_squared_error_reg])
    if set_.loss == 'loc_smooth_loss':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[loc_smooth_loss],
                      metrics=[mean_squared_error_reg])
    if set_.loss == 'laplacian_smooth':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[laplacian_smooth],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])

    # Euclidean Distance losses
    if set_.loss == 'ed_norm_sum2':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[ed_norm_sum2],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])
    if set_.loss == 'ed_border_sum':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[ed_border_sum],
                      metrics=[mean_squared_error_reg])
    if set_.loss == 'ed_smooth_sum':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[ed_smooth_sum],
                      metrics=[mean_squared_error_reg])
    if set_.loss == 'euclidean_distance':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[euclidean_distance],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])
    if set_.loss == 'ed_clip':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[ed_clip],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])
    if set_.loss == 'ed_sum2':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[ed_sum2],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])
    if set_.loss == 'ed_smooth':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[ed_smooth],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])
    # Intermediate Loss (paper = [0.005, 0.01, 0.02, 0.08, 0.32] / bad = [0.02, 0.05, 0.1, 0.2, 0.32])
    if set_.loss == 'ed_multi':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[ed_sum2, ed_sum2, ed_sum2, ed_sum2, ed_sum2],
                      loss_weights=[0.005, 0.01, 0.02, 0.08, 0.32],
                      metrics=['mse'])
    if set_.loss == 'euclidean_distance_siamese':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[siamese_euclidean],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])
    if set_.loss == 'ed_ae_sum':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[ed_ae_sum],
                      metrics=[mean_squared_error_reg])

    # Distance losses
    if set_.loss == 'manhattan_distance':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[manhattan_distance],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])
    if set_.loss == 'mse':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=['mse'],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])
    if set_.loss == 'mae':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=['mae'],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])

    # Classification losses
    if set_.loss == 'dice':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=[dice_coef_loss],
                      metrics=[dice_coef, 'binary_crossentropy', 'mse', 'accuracy'])

    if set_.loss == 'binary_xe':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=['binary_crossentropy'],
                      metrics=[dice_coef, 'mse', 'accuracy'])

    if set_.loss == 'xe':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=['categorical_crossentropy'],
                      metrics=[dice_coef, 'mse', 'accuracy'])

    if set_.loss == 'hinge':
        model.compile(optimizer=Adam(lr=set_.learning_rate),
                      loss=['hinge'],
                      metrics=[dice_coef, 'mse', 'accuracy'])

    # Model Checkpoint settings
    mdl_name = "loss_{val_loss:.6f}_" + weight_name
    model_checkpoint = ModelCheckpoint(mdl_name, monitor='val_loss', mode='min', save_best_only=True)

    # Adaptive Learning Rate
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min',
                                  patience=2, min_lr=1e-7, verbose=1)

    # Prepare Loss and Time History logging as Callback
    if set_.loss == 'ed_multi':
        loss_history = LossHistory_multi()
    elif set_.loss in ('ed_ae_sum', 'l1_loss', 'l2_loss', 'loc_smooth_loss', 'ed_border_sum', 'ed_smooth_sum', 'ed_irl_loss'):
        loss_history = LossHistory_reg()
    else:
        loss_history = LossHistory()

    # Get Model Summary / Plot Graph
    if set_.diag == 1:
        import pydot
        import graphviz
        print(model.summary())
        plot_model(model, show_shapes=True, to_file='model_test.pdf')

    # Batch Processing
    for e in range(0, set_.no_epochs):
        # Show current settings
        print('%' * 30)
        print('Parent Epoch:', e + 1, ' of', set_.no_epochs)
        print('%' * 30)
        show_setting_reduced(set_)

        # Create new shuffled list per epoch
        list_o = np.arange(0, start_valid)
        seedx.shuffle(list_o)

        for set_id, set_start in enumerate(set_test):
            print('-' * 30)
            print('Current Set:', set_id + 1, '/', set_.tot_set)
            set_stop = set_start + s_size
            # set_start = 0
            # set_stop = 1

            # Check batch-mode sampling
            if set_.batch_shuffle == 0:
                imgs_train, imgs_mask_train, imgs_id_train, flowxy_train = rd_hdf5_batch(dat.path_dataraw + dat.path_hdf5_scale, set_,
                                                                                         'p', [set_start, set_stop],
                                                                                         'range')
                imgs_valid, imgs_mask_valid, imgs_id_valid, flowxy_valid = rd_hdf5_batch(dat.path_dataraw + dat.path_hdf5_scale, set_,
                                                                                         'p',
                                                                                         [start_valid, stop_valid],
                                                                                         'range')

            if set_.batch_shuffle == 1:
                list_s = list_o[set_start: set_stop]
                imgs_train, imgs_mask_train, imgs_id_train, flowxy_train = rd_hdf5_batch(dat.path_dataraw + dat.path_hdf5_scale, set_,
                                                                                         'p', list_s, 'list')
                imgs_valid, imgs_mask_valid, imgs_id_valid, flowxy_valid = rd_hdf5_batch(dat.path_dataraw + dat.path_hdf5_scale, set_,
                                                                                         'p',
                                                                                         [start_valid, stop_valid],
                                                                                         'range')

            total_batch = imgs_train.shape[0]

            # DATA: image type conversion
            imgs_train = imgs_train.astype('float32')
            imgs_valid = imgs_valid.astype('float32')

            # DATA: image augmentation
            if set_.dataugm == 1:
                print('Data augmentation: in progress...')
                augmentlist = get_augmentationlist(set_, seedx, total_batch)
                imgs_train = get_augmentation_batch(set_, imgs_train, augmentlist)
                print('Data augmentation: done')

            # DATA: image scaling
            if set_.data_center == 0:
                placeholder = []
            if set_.data_center == 1:
                imgs_train /= 255.
                imgs_valid /= 255.
            if set_.data_center == 2:
                imgs_train /= 255.
                imgs_valid /= 255.
                imgs_train -= 0.5
                imgs_valid -= 0.5

            # DATA: image normalization
            if set_.data_normalize == 1:
                mean = np.mean(imgs_train)  # mean for data centering
                std = np.std(imgs_train)  # std for data normalization
                imgs_train -= mean
                imgs_train /= std
                imgs_valid -= mean
                imgs_valid /= std

            print('%' * 30)

            # DATA: label-mask scaling
            if set_.mdl_type in ('unet', 'siamese'):
                imgs_mask_train = imgs_mask_train.astype('float32')
                imgs_mask_train /= 255.  # scale masks to [0, 1]
                imgs_mask_valid = imgs_mask_valid.astype('float32')
                imgs_mask_valid /= 255.  # scale masks to [0, 1]

            # DATA: optical flow scaling
            if set_.mdl_type in (
                    'flownet1', 'flownet2','flownet4'):
                if set_.flow_scaled == 'pixel':
                    placeholder = []
                if set_.flow_scaled == 'pixel_pos':
                    flowxy_train[:, :, :, 0] += (set_.w_out)
                    flowxy_train[:, :, :, 1] += (set_.h_out)
                    flowxy_valid[:, :, :, 0] += (set_.w_out)
                    flowxy_valid[:, :, :, 1] += (set_.h_out)
                if set_.flow_scaled == 0:
                    flowxy_train[:, :, :, 0] /= (set_.w_out)
                    flowxy_train[:, :, :, 1] /= (set_.h_out)
                    flowxy_valid[:, :, :, 0] /= (set_.w_out)
                    flowxy_valid[:, :, :, 1] /= (set_.h_out)
                if set_.flow_scaled == 1:
                    flowxy_train[:, :, :, 0] /= (2 * set_.w_out)
                    flowxy_train[:, :, :, 1] /= (2 * set_.h_out)
                    flowxy_train += 0.5
                    flowxy_valid[:, :, :, 0] /= (2 * set_.w_out)
                    flowxy_valid[:, :, :, 1] /= (2 * set_.h_out)
                    flowxy_valid += 0.5
                if set_.flow_scaled == 2:
                    flowxy_train[:, :, :, 0] /= (set_.w_out)
                    flowxy_train[:, :, :, 1] /= (set_.h_out)
                    flowxy_train += 1
                    flowxy_valid[:, :, :, 0] /= (set_.w_out)
                    flowxy_valid[:, :, :, 1] /= (set_.h_out)
                    flowxy_valid += 1
                # [0, 8]
                if set_.flow_scaled == 8:
                    flowxy_train[:, :, :, 0] /= (set_.w_out / 4)
                    flowxy_train[:, :, :, 1] /= (set_.h_out / 4)
                    flowxy_train += 4
                    flowxy_valid[:, :, :, 0] /= (set_.w_out / 4)
                    flowxy_valid[:, :, :, 1] /= (set_.h_out / 4)
                    flowxy_valid += 4

                # DATA: Penalize non optical flow zones
                if set_.flow_penalize == 1:
                    if set_.flow_scaled == 0:
                        val = 0
                    if set_.flow_scaled == 1:
                        val = 0.5
                    flowxy_train[flowxy_train == val] = set_.flow_penalize_value
                    flowxy_valid[flowxy_valid == val] = set_.flow_penalize_value

                # Extend Outputvariable with FLOW, I1 + I2
                if set_.loss in ('ed_ae_sum', 'l1_loss', 'l2_loss', 'loc_smooth_loss', 'ed_border_sum', 'ed_smooth_sum', 'ed_irl_loss'):
                    sh_t = flowxy_train.shape
                    out_train = np.zeros((sh_t[0], sh_t[1]+6, sh_t[2], sh_t[3]), dtype=float)
                    out_train[:, :, :, 0:2] = flowxy_train
                    out_train[:, :, :, 2:7] = imgs_train
                    out_train[:, :, :, 8] = imgs_mask_train[:, :, :, 0]
                    sh_v = flowxy_valid.shape
                    out_valid = np.zeros((sh_v[0], sh_v[1]+3, sh_v[2], sh_v[3]), dtype=float)
                    out_valid[:, :, :, 0:2] = flowxy_valid
                    out_valid[:, :, :, 2:4] = imgs_valid
                    out_valid[:, :, :, 4] = imgs_mask_valid[:, :, :, 0]
                else:
                    out_train = flowxy_train
                    out_valid = flowxy_valid

            # TRAINING: Batch Training / Validation
            if set_.mdl_type == 'unet':
                if set_id != set_test.shape[0] - 1:
                    # Train Model during Sets (except last)
                    print('Train')
                    model.fit(imgs_train,
                              imgs_mask_train,
                              batch_size=set_.b_size,
                              verbose=1,
                              shuffle=True)
                else:
                    # Train + Validate on last Set of Epoch, save loss_history and best model (val-loss)
                    model.fit(imgs_train,
                              imgs_mask_train,
                              batch_size=set_.b_size,
                              verbose=1,
                              shuffle=True,
                              validation_data=[imgs_valid, imgs_mask_valid],
                              callbacks=[model_checkpoint, loss_history])


            if set_.mdl_type in ('flownet1', 'flownet2', 'flownet4'):
                if set_id != set_test.shape[0] - 1:
                    print('Train')
                    model.fit(imgs_train,
                              out_train,
                              batch_size=set_.b_size,
                              verbose=1,
                              shuffle=True)
                else:
                    model.fit(imgs_train,
                              out_train,
                              batch_size=set_.b_size,
                              verbose=1,
                              shuffle=True,
                              validation_data=[imgs_valid,
                                               out_valid],
                              callbacks=[model_checkpoint, loss_history])

            # Diagnose - Check scores from flownet
            if set_.diag == 1:
                y_pred = model.predict(imgs_train, verbose=1)
                y_true = out_train
                score_ed = K.eval(ed_clip(K.variable(y_true[:, :, :, 0:2]), K.variable(y_pred[:, :, :, :])))
                score_ed_norm_sum2 = K.eval(ed_norm_sum2(K.variable(y_true[:, :, :, 0:2]), K.variable(y_pred[:, :, :, :])))

def main():
    # Get initialization
    dat_train = get_train()
    set_train = get_set()
    # Check if re-train active, if so append to old history loss file
    if set_train.re_train == 0:
        csv_init(set_train, 'history_loss.csv')
    # Train
    fit_train_batch(dat_train, set_train)

if __name__ == '__main__':
   main()

