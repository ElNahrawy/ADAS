import numpy as np
import cv2

class ROI:
    def __init__(self, dims, contour):
        '''

        :param dims: frame dimensions
        :param contour: contour of the ROI (Region of interest)
        '''
        self.ROI = None
        self.height = dims[0]
        self.width = dims[1]
        self.contour = contour
        self.create_ROI()

    def create_ROI(self):
        self.ROI = np.zeros((self.height, self.width))
        cv2.fillPoly(self.ROI, pts=self.contour, color = (255,255,255))

    def visualize(self):
        cv2.imshow('ROI' ,self.ROI)
        cv2.waitKey(0)
        print(self.ROI.shape)

    def filter(self, classes, boxes):
        '''
        Discards all BBOXs out of the ROI
        :param classes: array of detected classes by the object detector
        :param boxes: array of bboxs detected by the object detector
        :return:
        '''
        num_obj = len(classes)
        indices = []

        for i in range(num_obj):
            left, top, width, height = boxes[i]
            if self.ROI[top:top+height, left:left+width].sum() == 0:
                indices.append(i)

        classes = np.delete(classes, indices)
        boxes = np.delete(boxes, indices, axis =0)

        return classes, boxes

    def fast_filter(self, classes, boxes):
        num_obj = len(classes)
        indices = []
        for i in range(num_obj):
            left, top, width, height = boxes[i]
            pH = int(round(top + height/2.0))
            pW = int(round(left + width/2.0))
            if self.ROI[pH, pW] == 0:
                indices.append(i)
        classes = np.delete(classes, indices)
        boxes = np.delete(boxes, indices, axis =0)
        return classes, boxes
