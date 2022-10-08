# ADAS 
## __Project Description__
One crucial aspect of safe driving is maintaining appropriate distance between the driver’s car and cars in front of it. Such behavior despite very important is not adapted by drivers enough. In this project we propose a smart system based on computer vision deep learning models, that detects different objects in unstructured roads environments while monitoring their distance from the used camera. This advanced driver assistant system alarms the driver when the intervehicle distance is below the safe distance.

## __System overview__
![System diagram](images\system_diagram.png)

The system is built upon:
* __Object detection__
    ![YOLO preview](images\yolo_preview.png)
     To detect different objects on the road. YOLOv4 was fine tuned on [IDD dataset](https://idd.insaan.iiit.ac.in/) to acheive the best generalization results in unstructured roads such as in Egypt. \
    Fine-tuned model results
    ![YOLO results](images\yolo_results.png)
* __Monocular depth estimation__

   Monocular depth estimation is used to measure the distance between the detected objects and the dashboard camera used.\
   We provide our own implementation and weights for [DenseDepth](https://github.com/ialhashim/DenseDepth) and [FastDepth](https://github.com/dwofk/fast-depth) trained on [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
   ![Depth estimation results](images\depth_results.png)

## __Trained Models__
* [DenseDepth Keras implementation](https://drive.google.com/file/d/1i_K86ZWAvmKYdUGwIAtiX3soDpMorxvT/view?usp=sharing)
* [FastDepth Keras Implementation](https://drive.google.com/file/d/1HnaiuD9PhrQwk_wXQO8zLeYEaX3Rvqw2/view?usp=sharing)
* [YOLOv4 fine tuned on IDD](https://drive.google.com/file/d/13bHboRFFeJ6hCJZEHmzLyCelIUuAB14B/view?usp=sharing)
* [Tiny YOLOv4 fine tuned on IDD](https://drive.google.com/file/d/1Ay5_Oh-8eZWZrfYG91bCmD37vwoyJkk-/view?usp=sharing)

## __Demo__
To run a quick demo install requirements file then run the inference script on your image/video\
**Image**\
`python inference_scripy.py -img -dir test.png`

**Video**\
`python inference_script.py -video -dir test.mp4`
## __Getting Started__

To run this POC on any image/video follow these steps
* Clone the repo \
``` git clone ```
* Download trained models into weights folder
* Install the requirements file
* Import MainClass and create an object with desired setup
```
from MainClass import MainClass
import numpy as np
app = MainClass(max_depth = 10,
                 use_ROI = False,
                 ROI_dims = (720, 1280),
                 ROI_points = np.array([[725,720], [475,720], [550, 300], [650, 300]]),
                 threshold = 3.5,
                 tiny_yolo = False,
                 yolo_size = 320,
                 depth_model = 'FastDepth',
                 use_gpu = False))
```

>   *__max_depth__*: maximun depth in the image, 10m for NYU depth v2 dataset, 80m for KITTI dataset\
   *__use_ROI__*: limits alerms to a certain region of the image determined by the ROI points\
   *__ROI_points__*: coordinates of the ROI(region of interest) that get set manually according to the dashboard camera in use\
   *__threshold__*: distance threshold for alarm\
   *__tiny_yolo__*: sets the use of the tiny YOLO model\
   *__yolo_size__*: sets input size for the YOLO model\
   *__depth_model__*: used to choose the used Depth model, either **FastDepth** or **DenseDepth**\
   *__use_gpu__*: sets the use of GPU, requires the installion of a version of OpenCV supporting GPU

* Run inference

   On Image\
` 
main.image_inference(source = 'dir', image_dir = 'test.jpg', scaling_factor = 0.8)
main.visualize(store_image = False, apply_on_depth = False, scaling_factor = 0.8, show_image=True)
`
 \
On video\
   `main.visualize_video(source = 'dir', dir = 'test.mp4', show_video=True, scaling_factor = 0.8, apply_on_depth = False, save_video=False) `

**For further details please read functions documentation supplied within the docstring.**
