import cv2
import os
import numpy as np
from tqdm import tqdm
import random

check_15_labels = False


def Regularise(data_array, mode):
    "Accepts a numpy array and normalises it by dividing by 255 and reducing mean. (custom mean if mode is train.)"
    total_blue = 0
    total_red = 0
    total_green = 0
    total_points = 0

    if mode == 'train':
        print('\nCalculating channel wise mean of {} data.'.format(mode))
        for sample in tqdm(data_array):
            for row in sample:
                for point in row:
                    total_blue += point[0]
                    total_green += point[1]
                    total_red += point[2]
                    total_points += 1
        mean_blue = int(total_blue / total_points)
        mean_green = int(total_green / total_points)
        mean_red = int(total_red / total_points)

    elif mode == 'val':
        mean_blue = 127
        mean_green = 127
        mean_red = 127
    else:
        print('Mujhe mode mein "train" ya "val" hi samajh aata hai.')
        return

    print('\nChannel wise reducing mean and normalising of {} data.'.format(mode))

    #data_array = data_array.astype(np.float64)
    for i in tqdm(range(len(data_array))):
        data_array[i] = data_array[i].astype(np.float64)
        data_array[i][:, :, 0] = data_array[i][:, :, 0] - mean_blue
        data_array[i][:, :, 1] = data_array[i][:, :, 1] - mean_green
        data_array[i][:, :, 2] = data_array[i][:, :, 2] - mean_red
        data_array[i] = data_array[i] / 255

    return(data_array)


def check_labels(img, locations):
    _, breadth, _ = img.shape
    points = [[0, 0], [0, 0], [0, int(breadth / 2)], [0, int(breadth / 2)], [0, breadth], [0, breadth]]
    c = 0
    for location in locations:
        points[c][0] = location
        c += 1
    cv2.line(img, (points[0][1] + 5, points[0][0]), (points[1][1] + 5, points[1][0]), (255, 0, 0), 5)
    cv2.line(img, (points[2][1], points[2][0]), (points[3][1], points[3][0]), (0, 255, 0), 5)
    cv2.line(img, (points[4][1] - 5, points[4][0]), (points[5][1] - 5, points[5][0]), (0, 0, 255), 5)

    cv2.imshow('1', cv2.resize(img, None, fx=700 / breadth, fy=700 / breadth))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fetch_data(data_path, resize_dimensions, mode):

    images = []
    labels = []

    cool_extensions = ('jpg', 'JPG')
    jpg_list = [file for file in os.listdir(data_path) if file.endswith(cool_extensions)]

    random_15 = list(range(len(jpg_list)))
    random.shuffle(random_15)
    random_15 = random_15[:15]

    counter = 0
    for file in jpg_list:
        naam = file.split('.')[0]
        print(naam)
        jpg_path = os.path.join(data_path, file)
        image = cv2.imread(jpg_path)
        txt_path = os.path.join(data_path, naam + '.txt')

        if not os.path.exists(txt_path):
            print('{} ke labels gum hain.'.format(naam))
            continue

        points = []
        with open(txt_path, 'r') as txt:
            for line in txt:
                points.append(int(line[:-1]))

        if counter in random_15 and check_15_labels:
            print('showing')
            check_labels(image, points)

        length, breadth, _ = image.shape
        re_length, re_breadth, _ = resize_dimensions

        fy = re_length / length
        fx = re_breadth / breadth
        #print(length, breadth, re_length, re_breadth, fy, fx)
        image = cv2.resize(image, None, fx=fx, fy=fy)
        points = [int(point * fy) for point in points]

        if counter in random_15 and check_15_labels:
            print('showing')
            check_labels(image, points)

        images.append(image)
        labels.append(points)

        counter += 1

    #images = np.array(images)
    #labels = np.array(labels)

    images = Regularise(data_array=images, mode=mode)
    print(np.unique(images))

    return(images, labels)


def Batchify(X, y, batch_size):

    number_samples = len(X)
    batched_X = []
    batched_y = []

    steps_num = (number_samples + batch_size - 1) // batch_size

    num_extras = number_samples % batch_size
    indices = list(range(number_samples))
    random.shuffle(indices)
    extra_indices = indices[:num_extras]
    for index in extra_indices:
        X.append(X[index])
        y.append(y[index])

    for step in range(steps_num):
        batch_X = X[step * batch_size:(step + 1) * batch_size]
        batch_y = y[step * batch_size:(step + 1) * batch_size]
        batched_X.append(batch_X)
        batched_y.append(batch_y)

    return(batched_X, batched_y, steps_num)
