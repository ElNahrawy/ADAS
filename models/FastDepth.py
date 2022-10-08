import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, UpSampling2D, ReLU, Concatenate, SeparableConv2D, Add, BatchNormalization, DepthwiseConv2D, Conv2D
from tensorflow.keras import Model
import keras.backend as K

def create_model():
    base_model = keras.applications.MobileNet(input_shape=(224, 224, 3), weights="imagenet", include_top=False)

    base_model_output_shape = base_model.layers[-1].output.shape
    for layer in base_model.layers: layer.trainable = True  # unfreezing densenet layers

    def upsampling(input_tensor, n_filters, skip_add = None):
        x = DepthwiseConv2D(kernel_size=(5, 5), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=n_filters/2, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        if skip_add is not None:
          x = Add()([x, skip_add])
        return x

    # layer name of encoders to be concatenated
    names = ['conv_pw_1_relu', 'conv_pw_3_relu', 'conv_pw_5_relu']
    # Decoder layers
    decoder = Conv2D(filters=1024, kernel_size=(7,7), padding='same', input_shape=base_model_output_shape)(base_model.output)  # bottel neck
    decoder = upsampling(decoder, 1024)
    decoder = upsampling(decoder, 512, base_model.get_layer(names[2]).output)
    decoder = upsampling(decoder, 256, base_model.get_layer(names[1]).output)
    decoder = upsampling(decoder, 128, base_model.get_layer(names[0]).output)
    decoder = upsampling(decoder, 64)

    # extract depths (final layer)
    conv3 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same')(decoder)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    
    out = Conv2D(filters=1, kernel_size=(1, 1), padding='same')(conv3)
    out = BatchNormalization()(out)
    out = ReLU()(out)

    # create the model
    model = Model(inputs=base_model.input, outputs=out)

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


