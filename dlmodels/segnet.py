import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from dlmodels.unet import conv, conv_block
from dlmodels.maxunpooling import MaxPoolingWithArgmax2D, MaxUnpooling2D


def up_sampling_block(x, indices, n_filters, batch_norm=True):
    y = conv(n_filters, kernel_size=1, use_bias=False)(x)
    y = MaxUnpooling2D(up_size=(2, 2))([y, indices])
    y = conv(n_filters)(y)
    if batch_norm:
        y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    return y


def segnet(input_shape, n_classes, batch_norm=True, dropout=0):
    inputs = layers.Input(input_shape)
        
    conv1 = conv_block(inputs, 32, batch_norm=batch_norm, dropout=dropout)
    pool1, indices1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128, batch_norm=batch_norm, dropout=dropout)
    pool2, indices2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256, batch_norm=batch_norm, dropout=dropout)
    pool3, indices3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512, batch_norm=batch_norm, dropout=dropout)
    pool4, indices4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 1024, batch_norm=batch_norm, dropout=dropout)

    up1 = up_sampling_block(conv5, indices4, 512)
    conv6 = conv_block(up1, 512, batch_norm=batch_norm, dropout=dropout)

    up2 = up_sampling_block(conv6, indices3, 256)
    conv7 = conv_block(up2, 256, batch_norm=batch_norm, dropout=dropout)

    up3 = up_sampling_block(conv7, indices2, 128)
    conv8 = conv_block(up3, 128, batch_norm=batch_norm, dropout=dropout)

    up4 = up_sampling_block(conv8, indices1, len(inputs) * 32)
    conv9 = conv_block(up4, 64, batch_norm=batch_norm, dropout=dropout)
    
    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(conv9)

    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model
