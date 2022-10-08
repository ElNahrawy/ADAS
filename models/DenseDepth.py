import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate
from tensorflow.keras import Model
import keras.backend as K

def create_model():
    base_model = keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet',
                                                         input_shape=(480, 640, 3))
    base_model_output_shape = base_model.layers[-1].output.shape
    for layer in base_model.layers: layer.trainable = True  # unfreezing densenet layers
    decode_filters = int(base_model_output_shape[-1])  # 1664

    def upsampling(input_tensor, n_filters, concat_layer):
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_tensor)
        x = Concatenate()([x, concat_layer])
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    # layer name of encoders to be concatenated
    names = ['pool3_pool', 'pool2_pool', 'pool1', 'conv1/relu']
    # Decoder layers
    decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape)(
        base_model.output)  # bottel neck
    decoder = upsampling(decoder, int(decode_filters / 2), base_model.get_layer(names[0]).output)
    decoder = upsampling(decoder, int(decode_filters / 4), base_model.get_layer(names[1]).output)
    decoder = upsampling(decoder, int(decode_filters / 8), base_model.get_layer(names[2]).output)
    decoder = upsampling(decoder, int(decode_filters / 16), base_model.get_layer(names[3]).output)

    # extract depths (final layer)
    conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(decoder)

    # create the model
    model = Model(inputs=base_model.input, outputs=conv3)

    return model


def depth_loss_function(y_true, y_pred):

  #Cosine distance loss
  l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
  
  # edge loss for sharp edges
  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
  
  # structural similarity loss
  l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)

  # weightage
  w1, w2, w3 = 1.0, 1.0, 0.1
  return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))


