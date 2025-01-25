import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def dk6(input_shape, n_classes):
    model = models.Sequential()
    model.add(layers.ZeroPadding2D((2, 2), input_shape=input_shape))
    model.add(layers.Conv2D(
              filters=16,
              kernel_size=(5, 5),
              dilation_rate=(1, 1)
    ))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.ZeroPadding2D((2, 2)))
    model.add(layers.MaxPooling2D(
              pool_size=(5, 5),
              strides=(1, 1)
    ))
    model.add(layers.ZeroPadding2D((4, 4)))
    model.add(layers.Conv2D(
              filters=32,
              kernel_size=(5, 5),
              dilation_rate=(2, 2)
    ))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.ZeroPadding2D((4, 4)))
    model.add(layers.MaxPooling2D(
            pool_size=(9, 9),
            strides=(1, 1)
    ))
    model.add(layers.ZeroPadding2D((6, 6)))
    model.add(layers.Conv2D(
              filters=32,
              kernel_size=(5, 5),
              dilation_rate=(3, 3)
    ))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.ZeroPadding2D((6, 6)))
    model.add(layers.MaxPooling2D(
            pool_size=(13, 13),
            strides=(1, 1)
    ))
    model.add(layers.ZeroPadding2D((8, 8)))
    model.add(layers.Conv2D(
              filters=32,
              kernel_size=(5, 5),
              dilation_rate=(4, 4)
    ))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.ZeroPadding2D((8, 8)))
    model.add(layers.MaxPooling2D(
            pool_size=(17, 17),
            strides=(1, 1)
    ))
    model.add(layers.ZeroPadding2D((10, 10)))
    model.add(layers.Conv2D(
              filters=32,
              kernel_size=(5, 5),
              dilation_rate=(5, 5)
    ))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.ZeroPadding2D((10, 10)))
    model.add(layers.MaxPooling2D(
            pool_size=(21, 21),
            strides=(1, 1)
    ))
    model.add(layers.ZeroPadding2D((12, 12)))
    model.add(layers.Conv2D(
              filters=32,
              kernel_size=(5, 5),
              dilation_rate=(6, 6)
    ))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.ZeroPadding2D((12, 12)))
    model.add(layers.MaxPooling2D(
            pool_size=(25, 25),
            strides=(1, 1)
    ))
    model.add(layers.Conv2D(
              filters=n_classes,
              kernel_size=(1, 1)
    ))
    model.add(layers.Activation(
              activation="softmax"
    ))
    return model
