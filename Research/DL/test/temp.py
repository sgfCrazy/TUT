# if not None:
#     print("sgf")
#
# # {"outputs": [{"image_path": "/home/appnfs2/sgf/jobs/data-center-20220721-151035-0x44fc/VOC/JPEGImages/000001.jpg",
# #               "boxes": [
# #                   ["1", 0.22455883026123047, [-785.92529296875, -550.35400390625, 664.07470703125, 575.42724609375]],
# #                   ["2", 0.23900270462036133, [-260.9375, -134.375, 126.5625, 146.875]],
# #                   ["8", 0.23851394653320312, [-251.55029296875, -365.631103515625, 117.19970703125, 378.118896484375]],
# #                   ["14", 0.23924708366394043, [-161.968994140625, -183.10546875, 27.581787109375, 195.60546875]],
# #                   ["15", 0.22281885147094727, [-2387.5244140625, -2026.55029296875, 2265.6005859375, 2051.57470703125]],
# #                   ["16", 0.24150657653808594,
# #                    [-86.2884521484375, -17.681884765625, -54.3548583984375, 23.895263671875]],
# #                   ["20", 0.22548675537109375, [-1036.73095703125, -1223.4375, 914.83154296875, 1248.4375]]]}]}
#
# s = ""
# if s is None:
#     print("sgfsgf")

from pathlib import *

# path = r"C:\Users\Songgf\Desktop\GIT\TUT\Research\DL\test\temp.py"
# t = Path(path)
# print()
# import sys
#
# print(__file__)
# print(sys.path)

import cv2

image_abspath = r'C:\Users\86158\Desktop\TUT\TUT\Research\DL\test\000001.jpg'
image = cv2.imread(image_abspath)

FONT_HERSHEY_SIMPLEX = 0
FONT_HERSHEY_PLAIN = 1
FONT_HERSHEY_DUPLEX = 2
FONT_HERSHEY_COMPLEX = 3
FONT_HERSHEY_TRIPLEX = 4
FONT_HERSHEY_COMPLEX_SMALL = 5
FONT_HERSHEY_SCRIPT_SIMPLEX = 6
FONT_HERSHEY_SCRIPT_COMPLEX = 7
font_scale = 1
thickness = 1
text = "sgf"
font = FONT_HERSHEY_TRIPLEX
text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)

xmin, ymin = 50, 50
xmax, ymax = 200, 200
color = (255, 0, 0)
cv2.rectangle(image, (xmin, ymin - text_size[1] - baseline), (xmin + text_size[0], ymin), color, -1)
# 速度：LINE_8>LINE_AA 美观：LINE_AA>LINE_8
cv2.putText(image, text, (xmin, ymin - baseline), font, font_scale, (255, 255, 255), thickness, lineType=8, bottomLeftOrigin=False)
cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, -1)
# cv2.imshow("image", image)
cv2.imwrite("image.jpg", image)
print()

# img = cv.putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType= 8, bottomLeftOrigin=False)
