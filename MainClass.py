from YOLO import YOLOV4
from Depth_FastDepth import Depth
from Depth_DenseDepth import Depth as Depth2
from ROI import ROI
import threading
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np


class MainClass:
    def __init__(self,camera = None, names = 'utils/classes.names', max_depth = 10,
                 use_ROI = True,
                 ROI_dims = (720, 1280),
                 ROI_points = np.array([[725,720], [475,720], [550, 300], [650, 300]]),
                 threshold = 3.5,
                 tiny_yolo = False,
                 yolo_size = 512,
                 depth_model = 'FastDepth',
                 use_gpu = False):
        '''

        :param camera: determines the camera index
        :param names: directory to the file containing classes names
        :param max_depth: maximum depth in the depth map (10 in case of NYU, 80 in case of KITTI)
        :param use_ROI: if true discard all bbox outside of the  ROI (Region Of Interest)
        :param ROI_dims: dimensions of the frame in use
        :param ROI_points: coordinates of the ROI measured from top left corner as origin
        :param threshold: maximum safe intervehicle distance
        :param tiny_yolo: if true use tiny yolo instead of full scale yolo
        :param yolo_size: yolo input size
        :param depth_model: if equal to 'FastDepth' use FastDepth model otherwise use DenseDepth
        :param use_gpu: if True use GPU else use CPU, requires installing a version of OpenCv supporting GPU
        '''
        if tiny_yolo:
            self.yolo = YOLOV4(cfg='utils/yolov4_tiny.cfg', weights='weights/yolov4-tiny.weights' ,input_size= yolo_size, use_gpu=use_gpu)
        else:
            self.yolo = YOLOV4(input_size=yolo_size,  use_gpu=use_gpu)
        if depth_model == 'FastDepth':
            self.depth = Depth()
        else:
            self.depth = Depth2()
        if camera is not None:
            self.camera = camera
        with open(names, 'r') as f:
            self.names = f.read().splitlines()
        self.depth_value = []
        self.max_depth = max_depth
        self.classes = None
        self.confidences = None
        self.boxes = None
        self.depth_map = None
        self.img = None
        self.ROI = ROI(ROI_dims, [ROI_points])
        self.use_ROI = use_ROI
        self.threshold = threshold/self.max_depth


    def __detection_routine__(self,img):
        '''
        :return: used to perform object detection inside of a thread
        updates self.classes, self.confidences, self.boxes with the detection values
        '''
        self.classes, self.confidences, self.boxes = self.yolo.return_boxes(img)

    def __depth_routine__(self,corrected_colors_image, true_dims = True):
        '''

        :param corrected_colors_image: RGB image
        :param true_dims: if True return the depth map in the original image dimensions otherwise
                         return depth map with the depth model output dimensions
        :return: used to perform monocular depth estimation inside of a thread
        updates self.depth_map with the estimated depth map of the frame

        '''
        self.depth_map = self.depth.inference(corrected_colors_image, true_dims=true_dims)

    def image_inference(self, source, image_dir='', image_array=None, scaling_factor=1):
        """
        MultiThreading is used for inference
        :param source: specifies the image source:
            'dir': an image stored in a specific directory
            'cam': a camera connected to the device
            'array': an array passed to the function
        :param image_dir: the image directory
        :param image_array: an array containing the image
        :param scaling_factor: scales down bbox with scaling_factor
        :return: Updates self.classes, self.confidences, self.boxes with YOLO inference result
                 Updates self.depth_map with the depth map

        """
        if source == 'dir':
            self.img = cv2.imread(image_dir)
        elif source == 'cam':
            camera = cv2.VideoCapture(self.camera)
            _, self.img = camera.read()
            camera.release()
        elif source == 'array':
            self.img = image_array

        detection_thread = threading.Thread(target=self.__detection_routine__, args=[self.img])
        corrected_colors_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        depth_thread = threading.Thread(target=self.__depth_routine__, args=[corrected_colors_image])
        depth_thread.start()
        detection_thread.start()

        depth_thread.join()
        detection_thread.join()

        if self.use_ROI:
            self.classes, self.boxes = self.ROI.filter(self.classes, self.boxes)

        self.get_depth(scaling_factor=scaling_factor)

    def scale_bbox(self, box, scaling_factor):
        '''

        :param box: yolo detection bounding box
        :param scaling_factor: scaling down factor
        :return: scales down box with scaling_factor
        '''
        left, top, width, height = box
        x = 1 - scaling_factor
        left, top, width, height = box
        left = left + x * width / 2
        left = round(int(left))
        top = top + x * height / 2
        top = int(round(top))
        width = width * scaling_factor
        width = int(round(width))
        height = height * scaling_factor
        height = int(round(height))
        return [left, top, width, height]

    def get_depth(self, scaling_factor=0.5):
        '''
        Calculates the average value of the depth of each detected object
        Depth value ranges from 0 to 255
        :param scaling_factor: bbox scaling down factor
        :return: Updates self.depth_value list with the average depth of each detected object
        '''
        self.depth_value = []
        if len(self.classes) > 0:
            for box in self.boxes:
                if scaling_factor == 1:
                    left, top, width, height = box
                else:
                    left, top, width, height = self.scale_bbox(box, scaling_factor)

                depth_value = self.depth_map[top :int(top + height),left:int(left+width)].mean()
                self.depth_value.append(depth_value)

    def visualize(self, store_image=False, show_image=False, return_image=False, apply_on_depth=False, scaling_factor=1):
        '''

        :param store_image: True for storing visualization result in out.png file
        :param show_image: True for showing the visualization results
        :param return_image: True returns the visualization results (used for video inference)
        :param apply_on_depth: Apply visualization on depth map
        :param scaling_factor: bbox scaling down factor
        :return: Visualization results if return_image is True
        '''

        boxes = self.boxes
        if apply_on_depth is False:
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img.astype('uint8')
        else:
            img = cv2.cvtColor(self.depth_map, cv2.COLOR_GRAY2BGR)
            plt.imsave('out1.png', img)
            img.astype('uint8')
        if len(self.classes) > 0:
            for classId, confidence, box, depth_value in zip(self.classes.flatten(),
                                                             self.confidences.flatten(),
                                                             boxes,
                                                             self.depth_value):

                if scaling_factor == 1:
                    left, top, width, height = box
                else:
                    left, top, width, height = self.scale_bbox(box, scaling_factor)
                if depth_value > self.threshold:
                    r = 0
                    g = 255
                    b = 0
                else:
                    r = 0
                    g = 0
                    b = 255

                label = f'{self.names[classId]}  {round(depth_value  * self.max_depth, 2)}'
                cv2.rectangle(img, (left, top), (left+width, top+height), color=(b, g, r), thickness=1)
                cv2.rectangle(img, (left, top), (left + len(label) * 7, top - 12), (b, g, r), cv2.FILLED)
                cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255 - b, 255 - g, 255 - r), 1,
                            cv2.LINE_AA)
        if show_image is True:
            cv2.namedWindow("Image_inference", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Image_inference", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Image_inference',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
            if store_image is True:
                plt.imsave('out.png', img.astype('uint8'))
        elif store_image is True:
            plt.imsave('out.png', img.astype('uint8'))
        if return_image is True:
            return img

    def visualize_video(self, source, dir='', show_video=False, save_video=False, scaling_factor=0.5, apply_on_depth=False):
        '''

        :param source: takes values 'dir': for video source being a directory
                                    'camera': for video source being a camera
        :param dir: directory of video if source is a directory
        :param show_video: if True shows the video
        :param save_video: if True saves the video in the cwd with name 'out.mp4'
        :param scaling_factor: bounding box scaling down factor
        :param apply_on_depth: رperforms visualization on the depth  map instead of the original video
        :return:
        '''
        if source == 'dir':
            video = cv2.VideoCapture(dir)
        elif source == 'camera':
            video = cv2.VideoCapture(self.camera)
        if save_video:
            frame_width = int(video.get(3))
            frame_height = int(video.get(4))
            out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                  (frame_width, frame_height))
        if show_video is False and save_video is False:
            return
        if not save_video:
            cv2.namedWindow("inference", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("inference", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                timer = time.time()
                self.image_inference(source='array', image_array=frame, scaling_factor=scaling_factor)
                timer = time.time() - timer
                print(f"time per frame: {timer} FPS: {1/timer}")
                frame = self.visualize(return_image=True, scaling_factor=scaling_factor, apply_on_depth=apply_on_depth)
                if show_video is True:
                    cv2.imshow('inference', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if save_video:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
