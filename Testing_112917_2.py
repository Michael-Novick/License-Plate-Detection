import os
import time
import numpy as np
print(np.__version__)
print(np.__path__)
import cv2
print('numpy, cv2 success')
# import sklearn.model_selection as model
import xml.etree.ElementTree as ET
import pandas as pd
# import scipy.linalg
# print('import linalg successful')
# # import scipy.ndimage
# print('import ndimage successful')
from skimage import transform as transform
print('import skimage transform successful')
print('Starting, preparing Keras . . . ' + time.asctime(time.localtime()))
import imutils
# import pydot
# import pydot_ng
# import graphviz
from keras.models import Sequential
from keras import optimizers
from keras import layers as layers
from keras import backend as K
from keras import regularizers
# from keras import utils as utils
# import h5py
#import matplotlib # .pyplot as plt
import json


def main():
    time.clock()
    current_dir = directory()
    training_label_dir = directory('training label directory')
    image_dir = directory('training image directory')

    # training_label_names = list_file_names(training_label_dir)
    training_image_names = list_file_names(image_dir)

    labels = retrieve_test_labels()

    training(current_dir, training_label_dir, image_dir, labels, training_image_names)
    return None


def directory(prompt=None):
    directory1 = os.path.dirname(os.path.abspath(__file__))
    if prompt == 'training label directory':
        directory1 = directory1 + '\Training_MyPictures_101517_Labels'
    elif prompt == 'training image directory':
        directory1 = directory1 + '\Training_MyPictures_101517'
    elif prompt is None:
        None
    else:
        directory1 = directory1 + input('Please indicate: ' + prompt + ': ')
    # if prompt is not None:
    #     directory1 = directory1 + input('Please indicate: ' + prompt + ': ')
    # print(directory1)
    return directory1


def list_file_names(dir):
    f = []
    filenames = next(os.walk(dir))[2]
    f.extend(filenames)
    # print(f)
    return f


def training(dir_c, dir_l, dir_i, labels, images):
    # train_labels, test_labels = generate_train_test(labels)

    window_shape = (64, 64, 3)
    print('Loading Model . . . ' + time.asctime(time.localtime()))
    cnn_model = create_CNN(window_shape)
    print('Begin Loop . . . ' + time.asctime(time.localtime()))
    test_loop(labels, dir_c, dir_i, dir_l, cnn_model, window_shape)
    return None


# def generate_train_test(labels):
#     train_labels, test_labels = model.train_test_split(labels, test_size = 0.82, shuffle = True)
#     # print(train_labels)
#     # print(test_labels)
#     f = open('test_110717_075.txt', 'w')
#     json.dump(test_labels, f)
#     f.close()
#     f = open('train_110717_025.txt', 'w')
#     json.dump(train_labels, f)
#     f.close()
#     return train_labels, test_labels


def test_loop(labels, dir_c, dir_i, dir_l, cnn_model, window_shape):
    # for test, make label = labels[0]
    # label = labels[0]
    area_window = window_shape[0] * window_shape[1]
    total_images = len(labels)
    total_iteration = 1

    plate_frames_found = 0
    plate_frames_total = 0
    false_frames = 0
    frames_total = 0

    labels = labels[0:12]
    for label in labels:  # [0:10]:
        image_data = retrieve_labels(label, dir_l)
        number_of_plates = image_data.shape[0]
        # print('number of plates')
        # print(number_of_plates)
        # print('')
        boxes_percent = np.empty((number_of_plates, 4))
        # print('boxes percent')
        # print(boxes_percent)
        # print('')
        # print(image_data.at[0, 'x_Percent'])
        for i in range(0, number_of_plates):
            # print(image_data.at[i, 'x_Percent'])
            # print('x percent '+str(i))
            # print(image_data.at[i, 'x_Percent'][1])
            boxes_percent[i][0] = image_data.at[i, 'y_Percent'][0]
            boxes_percent[i][1] = image_data.at[i, 'y_Percent'][1]
            boxes_percent[i][2] = image_data.at[i, 'x_Percent'][0]
            boxes_percent[i][3] = image_data.at[i, 'x_Percent'][1]
            # boxes_percent[i][0] = image_data.at[i, 'x_Percent'][0]
            # boxes_percent[i][1] = image_data.at[i, 'x_Percent'][1]
            # boxes_percent[i][2] = image_data.at[i, 'y_Percent'][0]
            # boxes_percent[i][3] = image_data.at[i, 'y_Percent'][1]
            # boxes_percent[i][0] = image_data.at[i, 'x_Percent'][0]
            # boxes_percent[i][1] = image_data.at[i, 'x_Percent'][1]
            # boxes_percent[i][2] = image_data.at[i, 'y_Percent'][0]
            # boxes_percent[i][3] = image_data.at[i, 'y_Percent'][1]
        # print('')
        # print('boxes percent')
        # print(boxes_percent)
        image, name = load_image(label, dir_i)
        # pyramid(image)
        pyr = tuple(transform.pyramid_gaussian(image, downscale=2, max_layer=3))
        shift_number = (24, 12, 8, 6, 2)
        iteration = 3
        for p in pyr[iteration:]:  # change back to 0:
            gen = sliding_window(p, shift_number[iteration], window_shape)
            # print('sliding window', iteration, location)
            # location = (0, 0, 0, 0)
            # print('p.shape')
            # print(p.shape)
            size = p.shape[0], p.shape[0], p.shape[1], p.shape[1]
            size = np.array(size)
            bndbox = np.multiply(boxes_percent, size)  # .T for transpose
            # print('bndbox')
            # print(bndbox)
            # print('')
            bndbox = bndbox.astype(np.int64)
            # print('bndbox')
            # print(bndbox)
            # print('')
            # bndbox = () #image_data()
            print('Testing - Image: ' + str(total_iteration) + '/' + str(total_images) + ' Size: '
                + str(iteration+1) + '/4    ' + time.asctime(time.localtime()))
            # image_iteration = 1
            for window, location in gen:
                # print('location'+str(location))
                # print('bndbox'+str(bndbox))
                # print('')
                # tag = 0
                y = np.asarray(0)
                y = y[np.newaxis, ...]
                for i in range(0, number_of_plates):
                    if location[0] <= bndbox[i][0] and location[1] >= bndbox[i][1] and location[2] <= bndbox[i][2] and location[3] >= bndbox[i][3]:
                        area_bndbox = (bndbox[i][1] - bndbox[i][0]) * (bndbox[i][3] - bndbox[i][2])
                        if area_bndbox >= 110:
                            y = np.asarray(1)
                            y = y[np.newaxis, ...]
                            # tag = 1
                            plate_frames_total += 1
                            # print('y')
                            # print(y)
                            # print('plate present')
                        else:
                            None
                            # print('plate too small')
                    else:
                        None
                        # print('no plate present')

                # ML Stuff ##
                # print(window)
                # model_input = np.concatenate((window, 1))
                # model_input = np.expand_dims(None, window)
                model_input = window[np.newaxis, ...]
                # if tag == 1:
                #     print('*** Plate should be present ***') #, batch_size=1)
                #     cv2.imshow('window',window)
                # testoutput = cnn_model.test_on_batch(model_input, y)
                testoutput = cnn_model.predict(model_input, batch_size=1, verbose=0)
                # testoutput = cnn_model.predict_on_batch(model_input)
                print(testoutput)
                # if testoutput[0][0] > testoutput[0][1]:
                #     print('**** Plate should be present ****')
                #     print('y = '+str(y))
                #     if tag == 1:
                #         plate_frames_found += 1
                #     else:
                #         false_frames += 1
                if testoutput > 0.30:
                    # print('y = '+str(y))
                    if y == 1:
                        print(str(frames_total) + '**** Correct ****')
                        plate_frames_found += 1
                    else:
                        print(str(frames_total) + '---- False ----')
                        false_frames += 1
                #     cv2.imshow('window', window)
                #     print(testoutput) #, batch_size=1)
                # cv2.waitKey(0)
                # cv2.destroyWindow('window')
                frames_total += 1
                # test_output = cnn_model.predict(model_input, batch_size=1, verbose=1)
                # print(test_output)#, batch_size=1)
                # cnn_model.train_on_batch([np.asarray(1), window[:,:,0], window[:,:,1], window[:,:,2]], np.asarray(y))  #, batch_size=1)
                # cv2.imshow('image'+str(iteration)+str(location[1]), window)
                # cv2.waitKey(0)
                # cv2.destroyWindow('image'+str(iteration)+str(location[1]))
                #  window, location = next(gen)
                # print(image_iteration)
                # image_iteration += 1
            iteration += 1
            # print(cnn_model.metrics_names)
        total_iteration += 1
        # cv2.imshow('image'+str(iteration)+str(location[1]), window)
        # cv2.waitKey(0)
        # cv2.destroyWindow('image'+str(iteration)+str(location[1]))
            # while location[1] != image.shape[0] and location[3] != image.shape[1]:
            #     # cv2.imshow('image'+str(iteration), p)
            #      window, location = next(gen)
            # iteration += 1
        # cv2.waitKey(0)
    # cnn_model.save_weights(dir_c + "\Keras_Models\Model_110717_01")
    print('DONE!!!   ' + time.asctime(time.localtime()))
    print('total frames')
    print(frames_total)
    print('plate frames found')
    print(plate_frames_found)
    print('plate frames total')
    print(plate_frames_total)
    print('false frame')
    print(false_frames)
    plate_detection_percentage = float(plate_frames_found/plate_frames_total)
    false_detection_percentage = float(false_frames/frames_total)
    print('plate detection percentage')
    print(plate_detection_percentage)
    print('false positive percentage')
    print(false_detection_percentage)

# For testing the reading of labels
# def train_loop(labels, dir_i):
#     #for label in labels:
#     label = labels[0]
#     image, name = load_image(label, dir_i)
#     cv2.imshow(label, image)
#     cv2.waitKey(0))


def load_image(label, dir):
    # print(label)
    label = label.rstrip('.xml') + '.jpg'
    # print(label)
    image_path = dir + '//' +label
    # print(image_path)
    image = cv2.imread(image_path)
    # print(image)
    return image, label


def retrieve_labels(label, dir):
    dir = dir + '//' + label
    # tree = ET.parse(dir)
    # root = tree.getroot()
    # print(root.tag)
    # print(root.attrib)
    # for child in root:
    #     print(child.tag, child.attrib)
    # print(root)
    # file = open(dir, mode='r', encoding='utf-8')
    # file.read()

    # tree = ET.parse(dir)
    # root = tree.getroot()
    # for child in root:
    #     print(child.tag, child.attrib)
    # print(root.tag, root.attrib)
    # print(root[4].attrib)
    tree = ET.parse(dir)
    root = tree.getroot()
    #print(root.findall('object'))
    #print(image_data)
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    c6 = []
    for data in root.findall('object'):
        plate_chars = data.find('name').text
        # print(plate_chars)
        c1.append(plate_chars)
        #box = np.empty([2, 2],dtype=int)
        #print(box)
        bndbox=data.find('bndbox')
        box_x = (int(bndbox.find('xmin').text), int(bndbox.find('xmax').text))
        box_y = (int(bndbox.find('ymin').text), int(bndbox.find('ymax').text))
        # print(box)
        c2.append(box_x)
        c3.append(box_y)
        #    image_data.append([box, plate_chars])
    #    print(image_data)
        # box[0][0] = int(bndbox.find('xmin').text)
        # box[0][1] = int(bndbox.find('ymin').text)
        # box[1][0] = int(bndbox.find('xmax').text)
        # box[1][1] = int(bndbox.find('ymax').text)
        # # box[0][0]=next(bndbox)
        # box[0][1]=next(bndbox)
        # box[1][0]=next(bndbox)
        # box[1][1]=next(bndbox)
        # print(image_size_1)
        # xmin, xmax, ymin, ymax = box.find('xmin'), box.find('xmax'), box.find('ymin'), box.find('ymax')
        # print(xmin, xmax, ymin, ymax)
        # name = box.get('name')
        # print(name)
    image_size = root.find('size')
    image_size_1 = (int(image_size.find('width').text), int(image_size.find('height').text))
    c4 = [image_size_1]*len(c1)
    iter = 0
    for item in c2:
        c5.append((float(item[0]) / image_size_1[0], float(item[1]) / image_size_1[0]))
        c6.append((float(c3[iter][0]) / image_size_1[1], float(c3[iter][1]) / image_size_1[1]))
        iter = iter + 1
    image_data = pd.DataFrame({'Plate' : c1, 'Bounding_Box_x' : c2, 'Bounding_Box_y' : c3, 'Image Size' : c4, 'x_Percent': c5, 'y_Percent' : c6})
    # image_data.
    # print(image_data)
    # cv2.waitKey(0)
    return image_data


def sliding_window(image, shift_number, window_shape):
    rows, cols, dim = image.shape
    window_shape = window_shape
    for r in range(0, rows - window_shape[0], shift_number):
        for c in range(0, cols- window_shape[1], shift_number):
            bounds = (r, r+window_shape[0], c, c+window_shape[1])
            new_image = image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            # cv2.imshow(str(c),new_image)
            # cv2.waitKey(0)
            # cv2.destroyWindow(str(c))
            # print('AAAA')
            yield new_image, bounds
        bounds = (r, r+window_shape[0], cols-window_shape[1], cols)
        new_image = image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
        # cv2.imshow(str(c), new_image)
        # cv2.waitKey(0)
        # print('BBBBBB')
        yield new_image, bounds
    bounds = (rows-window_shape[0], rows, 0, window_shape[1])
    # bounds[0], bounds[1] = rows-window_shape[0], rows
    for c in range(0, cols - window_shape[1], shift_number):
        bounds = (rows-window_shape[0], rows, c, c + window_shape[1])
        # bounds[2], bounds[3] = c, c + window_shape[1]
        new_image = image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
        # cv2.imshow(str(c), new_image)
        # cv2.waitKey(0)
        # print('CCCCC')
        yield new_image, bounds
    bounds =(rows-window_shape[0], rows, cols - window_shape[1], cols)
    # bounds[2], bounds[3] = cols - window_shape[1], cols
    new_image = image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
    # cv2.imshow(str(c), new_image)
    # cv2.waitKey(0)
    # print('DDDD')
    return new_image, bounds

    #shift_number = 20
    # i_row = 0
    # row_new = (0, window_shape[0])
    # col_new = (0, window_shape[1])
    # while row_new[1] <= rows && col_new[1] <= cols:
    #     while row_new[1] <= rows:
    #         i_col = 0
    #         while col_new[1] <= cols:
    #             while True:
    #                 if :
    #                     return None, None
    #                 else:
    #                     yield position, new_image
    #         i_row += 1
    #         row_new = (row_new[0]+i_row*shift_number, row_new[1]+i_row*shift_number)


def pyramid(image):
    pyr = tuple(transform.pyramid_gaussian(image, downscale=2, max_layer=4))
    i_row = 0
    for p in pyr[0:]:
        cv2.imshow('image'+str(i_row),p)
        i_row += 1
        # n_rows, n_cols = p.shape[:2]
        # composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        # i_row += n_rows
        # image_1 = imutils.resize(composite_image, width=800)
        # cv2.imshow('image'+str(i_row-1),image_1)
    # fig, ax = matplotlib.pyplot.subplots()
    # ax.imshow(composite_image)
    # matplotlib.pyplot.show()
    cv2.waitKey(0)


def create_CNN(window_shape):
    K.set_image_dim_ordering('tf')
    # sgd = optimizers.SGD()
    adam = optimizers.Adam(lr=0.001, decay=0.000001)
    cnn_model = Sequential()
    print(window_shape)
    cnn_model.add(layers.ZeroPadding2D(padding=2, input_shape=window_shape, data_format='channels_last'))
    cnn_model.add(
        layers.Conv2D(12, (5, 5), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2()))  # 'relu'))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    cnn_model.add(layers.ZeroPadding2D(padding=(1, 1), data_format=None))
    cnn_model.add(layers.Conv2D(24, (3, 3), strides=(1, 1), activation='sigmoid',
                                kernel_regularizer=regularizers.l2()))  # 'relu')) #4 previously
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(20, activation='relu',
                               kernel_regularizer=regularizers.l2()))  # 'sigmoid'))  # layers.Dropout(rate, noise_shape=None, seed=None)
    # cnn_model.add(layers.Dropout(0.5))
    # cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(1, activation='sigmoid',
                               kernel_regularizer=regularizers.l2()))  # layers.Dropout(rate, noise_shape=None, seed=None)
    # cnn_model.add(layers.Dropout(0.1))
    # cnn_model.add(layers.Activation('softmax'))
    cnn_model.summary()
    # cnn_model.add(layers.Dense())
    cnn_model.compile(optimizer=adam, loss='binary_crossentropy')
    cnn_model.load_weights('Keras_Models_Saved\Model_112917_2220_binary')
    # utils.plot_model(cnn_model, to_file='model_1107.png', show_shapes=True, show_layer_names=True, rankdir='LR')
    print(cnn_model.get_weights())
    return cnn_model


def retrieve_test_labels():
    labels = json.load(open('test_112917_2220_binary.txt','r'))
    # print(labels[4])
    # cv2.waitKey(0)
    # labels = [labels[0]]
    return labels


if __name__ == '__main__':
    main()