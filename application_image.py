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
main.image_inference(source = 'dir', image_dir = 'E:\GP\Pycharm_project\Egyptian_police_vehicle_in_Aswan.jpg', scaling_factor = 0.8)
main.visualize(store_image = False, apply_on_depth = False, scaling_factor = 0.8, show_image=True)