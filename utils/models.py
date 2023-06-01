

import tensorflow as tf

from keras.layers import UpSampling2D, BatchNormalization, Activation, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Conv2DTranspose
from tensorflow.keras.layers import concatenate, average
from keras.models import Model, Sequential
from keras import backend as K
from keras.layers import ZeroPadding2D, Layer, InputSpec

#K.set_image_data_format('channels_first')       # Theano dimension ordering
K.set_image_data_format('channels_last')       # Theano dimension ordering


# Extending the ZeroPadding2D layer to do reflection padding instead.
class SymmetricPadding2D(ZeroPadding2D):
    def call(self, x, mask=None):
        pattern = [[0, 0],
                   [self.top_pad, self.bottom_pad],
                   [self.left_pad, self.right_pad],
                   [0, 0]]
        return tf.pad(x, pattern, mode='SYMMETRIC')


class MirrorPadding2D(Layer):
    '''Zero-padding layer for 2D input (e.g. picture).

    # Arguments
        padding: tuple of int (length 2), or tuple of int (length 4), or dictionary.
            - If tuple of int (length 2):
            How many zeros to add at the beginning and end of
            the 2 padding dimensions (rows and cols).
            - If tuple of int (length 4):
            How many zeros to add at the beginning and at the end of
            the 2 padding dimensions (rows and cols), in the order
            '(top_pad, bottom_pad, left_pad, right_pad)'.
            - If dictionary: should contain the keys
            {'top_pad', 'bottom_pad', 'left_pad', 'right_pad'}.
            If any key is missing, default value of 0 will be used for the missing key.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, channels, padded_rows, padded_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, padded_rows, padded_cols, channels)` if dim_ordering='tf'.
    '''

    def __init__(self,
                 padding=(1, 1),
                 dim_ordering='default',
                 **kwargs):
        super(MirrorPadding2D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.padding = padding
        if isinstance(padding, dict):
            if set(padding.keys()) <= {'top_pad', 'bottom_pad', 'left_pad', 'right_pad'}:
                self.top_pad = padding.get('top_pad', 0)
                self.bottom_pad = padding.get('bottom_pad', 0)
                self.left_pad = padding.get('left_pad', 0)
                self.right_pad = padding.get('right_pad', 0)
            else:
                raise ValueError('Unexpected key found in `padding` dictionary. '
                                 'Keys have to be in {"top_pad", "bottom_pad", '
                                 '"left_pad", "right_pad"}.'
                                 'Found: ' + str(padding.keys()))
        else:
            padding = tuple(padding)
            if len(padding) == 2:
                self.top_pad = padding[0]
                self.bottom_pad = padding[0]
                self.left_pad = padding[1]
                self.right_pad = padding[1]
            elif len(padding) == 4:
                self.top_pad = padding[0]
                self.bottom_pad = padding[1]
                self.left_pad = padding[2]
                self.right_pad = padding[3]
            else:
                raise TypeError('`padding` should be tuple of int '
                                'of length 2 or 4, or dict. '
                                'Found: ' + str(padding))

        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering must be in {tf, th}.')
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2] + self.top_pad + self.bottom_pad if input_shape[2] is not None else None
            cols = input_shape[3] + self.left_pad + self.right_pad if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    rows,
                    cols)
        elif self.dim_ordering == 'tf':
            rows = input_shape[1] + self.top_pad + self.bottom_pad if input_shape[1] is not None else None
            cols = input_shape[2] + self.left_pad + self.right_pad if input_shape[2] is not None else None
            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

    def call(self, x, mask=None):
        return self.asymmetric_spatial_2d_padding(x,
                                                  top_pad=self.top_pad,
                                                  bottom_pad=self.bottom_pad,
                                                  left_pad=self.left_pad,
                                                  right_pad=self.right_pad,
                                                  dim_ordering=self.dim_ordering)

    def asymmetric_spatial_2d_padding(self, x, top_pad=1, bottom_pad=1,
                                      left_pad=1, right_pad=1,
                                      dim_ordering='default'):
        '''Pad the rows and columns of a 4D tensor
        with "top_pad", "bottom_pad", "left_pad", "right_pad" (resp.) zeros
        rows on top, bottom; cols on left, right.
        '''
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if dim_ordering not in {'th', 'tf'}:
            raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

        if dim_ordering == 'th':
            pattern = [[0, 0],
                       [0, 0],
                       [top_pad, bottom_pad],
                       [left_pad, right_pad]]
        else:
            pattern = [[0, 0],
                       [top_pad, bottom_pad],
                       [left_pad, right_pad],
                       [0, 0]]
        return tf.pad(x, pattern, mode="SYMMETRIC")

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(MirrorPadding2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# Classic - UNet
def get_unet(set):
    # new
    inputs = Input((set.ch_img, set.h_out, set.w_out))

    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Connection Layer Down -> Up
    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool4)
    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv5)

    # Upscaling
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(up6)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu')(up7)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(up8)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu')(up9)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model



# FlowNet
def get_flownet(set_):
    # new
    inputs = Input((set_.ch_img, set_.h_out, set_.w_out))

    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Connection Layer Down -> Up
    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool4)
    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv5)

    # Upscaling
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(up6)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu')(up7)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(up8)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu')(up9)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv9)

    output1 = Conv2D(2, (1, 1), activation='tanh')(conv9)

    model = Model(inputs=inputs, outputs=output1)
    return model


def get_flownet2(set_):
    # new
    inputs = Input((set_.ch_img, set_.h_out, set_.w_out))

    conv1 = Conv2D(32, (7, 7), padding='same', activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1_1 = Conv2D(32, (7, 7), padding='same', activation='relu')(pool1)

    conv2 = Conv2D(64, (5, 5), padding='same', activation='relu')(conv1_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2_1 = Conv2D(128, (5, 5), padding='same', activation='relu')(pool2)

    conv3 = Conv2D(128, (5, 5), padding='same', activation='relu')(conv2_1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool3)

    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3_1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool4)

    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv4_1)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool5)
    pool5_1 = MaxPooling2D(pool_size=(2, 2))(conv5_1)

    conv6 = Conv2D(1024, (3, 3), padding='same', activation='relu')(pool5_1)


    # Upscaling
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5_1], axis=1)
    flow6 = Conv2D(512, (3, 3), padding='same', activation='relu')(up5)

    up4 = concatenate([UpSampling2D(size=(2, 2))(flow6), conv4_1], axis=1)
    flow5 = Conv2D(256, (3, 3), padding='same', activation='relu')(up4)

    up3 = concatenate([UpSampling2D(size=(2, 2))(flow5), conv3_1], axis=1)
    flow4 = Conv2D(128, (5, 5), padding='same', activation='relu')(up3)

    up2 = concatenate([UpSampling2D(size=(2, 2))(flow4), conv2_1], axis=1)
    flow3 = Conv2D(64, (5, 5), padding='same', activation='relu')(up2)

    up1 = concatenate([UpSampling2D(size=(2, 2))(flow3), conv1_1], axis=1)
    flow2 = Conv2D(32, (5, 5), padding='same', activation='relu')(up1)

    up0 = concatenate([UpSampling2D(size=(2, 2))(flow2), conv1], axis=1)
    flow1 = Conv2D(32, (5, 5), padding='same', activation='relu')(up0)

    output1 = Conv2D(2, (3, 3), activation='tanh', padding='same')(flow1)

    model = Model(inputs=inputs, outputs=output1)
    return model


# --------------------------------------------------------
# FlowNet
def get_flownet4(set_):
    # Settings
    pb_drop = 0.0
    inputs = Input((set_.ch_img, set_.h_out, set_.w_out))

    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(inputs)
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(pb_drop)(pool1)

    conv2 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(drop1)
    conv2 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(pb_drop)(pool2)

    conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(drop2)
    conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(pb_drop)(pool3)

    conv4 = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(drop3)
    conv4 = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(pb_drop)(pool4)

    # Connection Layer Down -> Up
    conv5 = Conv2D(1024, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(drop4)
    conv5 = Conv2D(1024, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(conv5)

    # Upscaling
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(up6)
    conv6 = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(up7)
    conv7 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(up8)
    conv8 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(up9)
    conv9 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=set_.initializer)(conv9)

    output1 = Conv2D(2, (3, 3), padding='same', activation='linear')(conv9)
    model = Model(inputs=inputs, outputs=output1)
    return model