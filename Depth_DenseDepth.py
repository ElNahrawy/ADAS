
from models.DenseDepth import create_model
import cv2
import tensorflow as tf
import numpy as np
class Depth():
    def __init__(self, weights='weights/Model_epoch_45-val_loss_0.04189-train_loss_0.03290.hdf5'):
        '''

        :param weights: Path to the model's weights
        '''
        self.model = create_model()
        self.model.load_weights(weights)
        self.original_dims = None
        self.im = None

    def load(self, image, dims):
        '''

        :param image: Image's path or the image as an np array
        :param dims: Dimensions for the image to be resized to
        :return:
        '''
        if isinstance(image, str):
            self.im = cv2.imread(image)
            self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)

        elif isinstance(image, np.ndarray):
            self.im = image
        assert(isinstance(self.im, np.ndarray))
        self.im = self.im / 255.0
        self.original_dims = self.im.shape[0:2]
        self.im = np.expand_dims(self.im, 0)
        self.im = tf.image.resize(self.im, dims)

    def inference(self, image, dims = (480,640), true_dims = False):
        '''

        :param image: Image's path or the image as an np array
        :param dims: Dimensions for the image to be resized to so it gets compatible with the depth model input layer
        :param true_dims: if True returns the inference results with the same dimensions as the original image
        :return: 2D array containing depth value for each pixel of the image
        '''
        self.load(image, dims)
        out = self.model.predict(self.im)
        if true_dims:
            return cv2.resize(out[0,:,:,:], (self.original_dims[1], self.original_dims[0]))
        return out[0,:,:,0]
