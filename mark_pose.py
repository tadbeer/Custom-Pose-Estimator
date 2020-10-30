"""
Displays the images in given folder one by one, and user marks the vertices of a polygon on it,
by right clicking on them.
The quadilateral bound inside thw four points shall be extracted out and warped to a reactangular shape.
The order of marking the four points shall be beginning from top left and moving clockswise toward bottom left.
Saves a dictionary of image name and its four pints in the above order as a pickle file.
Saves the picked quadilateral in the other specified folder.
"""

import urllib
import cv2
import numpy as np
import pickle
import os

#from pickquad import PickQuad

path = 'images'

cool_extensions = ('png', 'PNG', 'jpg', 'JPG', 'tiff', 'tif', 'TIFF', 'TIF', 'jpeg', 'JPEG')

file_names = os.listdir(path)
file_names = [k for k in file_names if k.endswith(cool_extensions)]
# print(file_names)

vertices = {}

for file in file_names:
    vertices[file] = []


def get_rightclicks(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global vertices
        global counter_clicks
        global file_name
        vertices[file_name].append(y)
        counter_clicks += 1


image_count = 0
while image_count < len(file_names):
    # print(vertices)
    file_name = file_names[image_count]
    file_naam = file_name.split('.')[0]
    file_path = os.path.join(path, file_name)
    img = cv2.imread(file_path)
    l, b, _ = img.shape
    b = int(b / 2)

    counter_clicks = 0

    #scale_width = 640 / img.shape[1]
    scale_height = 800 / img.shape[0]
    #scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale_height)
    window_height = int(img.shape[0] * scale_height)
    img = cv2.line(img, (b, 0), (b, l), (0, 0, 255), 5)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)

    cv2.setMouseCallback('image', get_rightclicks)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print(vertices)
    image_vertices = vertices[file_name]
    # print(image_vertices)

    if len(image_vertices) != 6:
        print('{} mein, {} points kaahe mar kiye be!'.format(file_naam, len(image_vertices)))
        vertices[file_name] = []
        continue
    else:
        image_count += 1
        txt_path = os.path.join(path, '{}.txt'.format(file_naam))
        print(file_naam, image_vertices)
        with open(txt_path, 'w') as file:
            for point in image_vertices:
                file.write('{}\n'.format(point))


# print(vertices)
# with open('new_points.pickle', 'wb') as file:
    #pickle.dump(vertices, file, protocol=pickle.HIGHEST_PROTOCOL)

"""
with open('four_points.pickle', 'rb') as handle:
    four_points = pickle.load(handle)
"""

'''for file in four_points:
    file_path = path + '/' + file
    img = cv2.imread(file_path)
    naam = file.split('\\')[-1].split('.')[0]
    print(naam)

    points = four_points[file]
    points = points.astype('int32')
    crop_im = PickQuad(img, points[0], points[1], points[2], points[3])
    cv2.imwrite('crops/{}.jpg'.format(naam), crop_im)
'''
