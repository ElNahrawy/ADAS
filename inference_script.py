from MainClass import MainClass
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-img', action='store_true', help='to run on image')
parser.add_argument('-video', action='store_true', help='to run on video')
parser.add_argument('-dir', type=str, help='image/video directory')
args = parser.parse_args()

main = MainClass(max_depth=10,
                 use_ROI=False,
                 ROI_dims=(720, 1280),
                 ROI_points=np.array([[725, 720], [475, 720], [550, 300], [650, 300]]),
                 threshold=3.5,
                 tiny_yolo=False,
                 yolo_size=320,
                 depth_model='FastDepth')
if args.img and args.video:
    print("Error choose only one type file or video")
elif args.img:
    main.image_inference(source='dir', image_dir=args.dir, scaling_factor=1)
    main.visualize(store_image=True, apply_on_depth=False, scaling_factor=1, show_image=True)
elif args.video:
    main.visualize_video(source='dir', dir=args.dir, show_video=True,
                         scaling_factor=1, apply_on_depth=True)