from MainClass import MainClass
import numpy as np
main = MainClass(max_depth = 10,
                 use_ROI = False,
                 ROI_dims = (720, 1280),
                 ROI_points = np.array([[725,720], [475,720], [550, 300], [650, 300]]),
                 threshold = 3.5,
                 tiny_yolo = True,
                 yolo_size = 320,
                 depth_model = 'FastDepth')
main.visualize_video(source = 'dir', dir = 'E:\GP\Pycharm_project\Final_Project\\test.mp4', show_video=True, scaling_factor = 0.8, apply_on_depth = True)
