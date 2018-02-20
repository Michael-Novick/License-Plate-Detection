import os
import time
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
from skimage import transform as transform
print('Starting, preparing Keras . . . ' + time.asctime(time.localtime()))
from keras.models import Sequential
from keras import optimizers
from keras import regularizers
from keras import layers as layers
from keras import backend as K
import json
import imutils


def main():
    time.clock()
    current_dir = directory()
    image_dir = "C://Users//User//Pictures//Image"
    # image_dir = "C://Users//User//Pictures//Camera Roll"
    images = list_file_names(image_dir)
    images = images[-1:]
    print(images)
    training(current_dir, image_dir, images)
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


def training(dir_c, dir_i, images):
    window_shape = (64, 64, 3)
    print('Loading Model . . . ' + time.asctime(time.localtime()))
    cnn_model = create_CNN(window_shape)
    print('Done . . . ' + time.asctime(time.localtime()))
    test_loop(dir_c, dir_i, cnn_model, window_shape, images)
    return None


def test_loop(dir_c, dir_i, cnn_model, window_shape, images):
    total_images = len(images)
    total_iteration = 1
    scale = 2
    plate_frames_found = 0
    for image in images:
        plate_list = []
        scale_list = []
        pic, name = load_image(image, dir_i)
        # pyramid(image)
        pic_smaller = imutils.resize(pic, width=1366, height=768)
        # pic_smaller = cv2.bilateralFilter(pic_smaller, 9, 75, 75)
        # cv2.imshow('Image', pic_smaller)
        # cv2.waitKey(3000)
        pyr = tuple(transform.pyramid_gaussian(pic_smaller, downscale=scale, max_layer=3))
        # shift_number = (12, 6, 4, 2, 1)
        shift_number = (24, 12, 8, 6, 2)
        iteration = 0
        for p in pyr[0:]:  # change back to 0:
        # for p in pyr[iteration:]:  # change back to 0:
            gen = sliding_window(p, shift_number[iteration], window_shape)
            size = p.shape[0], p.shape[0], p.shape[1], p.shape[1]
            size = np.array(size)
            print('Testing - Image: ' + str(total_iteration) + '/' + str(total_images) + ' Size: '
                + str(iteration+1) + '/4    ' + time.asctime(time.localtime()))
            for window, location in gen:
                # print(location)
                bounding_rect = np.asarray(location) * int(scale ** iteration)
                display_image = pic_smaller.copy()
                cv2.rectangle(display_image, (bounding_rect[3], bounding_rect[1]), (bounding_rect[2], bounding_rect[0]),
                              (255, 0, 0), thickness=3)
                # display_image = imutils.resize(display_image, width=1366, height=768)
                cv2.imshow('Input Image', display_image)
                model_input = window[np.newaxis, ...]
                testoutput = cnn_model.predict(model_input, batch_size=1, verbose=0)
                print(testoutput)
                if testoutput > 0.53:
                    plate_list.append(location)
                    scale_list.append(iteration)
                cv2.waitKey(1)
                # cv2.destroyWindow('Input Image')
            iteration +=1
        # print(plate_list)
        # print(scale_list)
        # cv2.destroyWindow('Input Image')
        new_image = display_plates(pic_smaller, plate_list, scale_list, scale)
        imutils.resize(new_image, width=1366, height=768)
        cv2.imshow('Plates', new_image)
        cv2.waitKey(0)
        cv2.destroyWindow('Plates')
        # cv2.destroyWindow('Image')
    print('DONE!!!   ' + time.asctime(time.localtime()))


def display_plates(pic, plate_list, scale_list, scale):
    for i in range(0, len(plate_list)):
        bounding_rect = np.asarray(plate_list[i]) * int(scale**scale_list[i])
        cv2.rectangle(pic, (bounding_rect[3], bounding_rect[1]), (bounding_rect[2], bounding_rect[0]), (0, 0, 255), thickness=1)
    return pic


def load_image(label, dir):
    #     # print(label)
    #     label = label.rstrip('.xml') + '.jpg'
    #     # print(label)
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
    adam = optimizers.Adam(lr=0.0000001)  #  , decay=0.000001)
    cnn_model = Sequential()
    print(window_shape)
    cnn_model.add(layers.ZeroPadding2D(padding=2, input_shape=window_shape, data_format='channels_last'))
    cnn_model.add(
        layers.Conv2D(18, (5, 5), strides=(1, 1), activation='relu'))  #, kernel_regularizer=regularizers.l2()))  # 'relu'))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    cnn_model.add(layers.ZeroPadding2D(padding=(1, 1), data_format=None))
    cnn_model.add(layers.Conv2D(36, (3, 3), strides=(1, 1), activation='relu')) #,kernel_regularizer=regularizers.l2()))  # 'relu')) #4 previously
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(20, activation='relu'))  #, kernel_regularizer=regularizers.l2()))  # 'sigmoid'))  # layers.Dropout(rate, noise_shape=None, seed=None)
    cnn_model.add(layers.Dropout(0.5))
    # cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(1, activation='sigmoid'))  #,kernel_regularizer=regularizers.l2()))  # layers.Dropout(rate, noise_shape=None, seed=None)
    # cnn_model.add(layers.Dropout(0.1))
    # cnn_model.add(layers.Activation('softmax'))
    cnn_model.summary()
    # cnn_model.add(layers.Dense())
    cnn_model.compile(optimizer=adam, loss='binary_crossentropy')
    # K.set_image_dim_ordering('tf')
    # # sgd = optimizers.SGD()
    # adam = optimizers.Adam(lr=0.0001)
    # cnn_model = Sequential()
    # print(window_shape)
    # cnn_model.add(layers.ZeroPadding2D(padding=2, input_shape=window_shape, data_format='channels_last'))
    # cnn_model.add(
    #     layers.Conv2D(12, (5, 5), strides=(1, 1), activation=None, kernel_regularizer=regularizers.l2()))  # 'relu'))
    # cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # cnn_model.add(layers.ZeroPadding2D(padding=(1, 1), data_format=None))
    # cnn_model.add(layers.Conv2D(8, (3, 3), strides=(1, 1), activation=None,
    #                             kernel_regularizer=regularizers.l2()))  # 'relu')) #4 previously
    # cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # cnn_model.add(layers.Flatten())
    # cnn_model.add(layers.Dense(20, activation=None,
    #                            kernel_regularizer=regularizers.l2()))  # 'sigmoid'))  # layers.Dropout(rate, noise_shape=None, seed=None)
    # # cnn_model.add(layers.Dropout(0.2))
    # # cnn_model.add(layers.Flatten())
    # cnn_model.add(layers.Dense(2, activation='softmax', kernel_regularizer=regularizers.l2()))
    # cnn_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


    # K.set_image_dim_ordering('tf')
    # sgd = optimizers.SGD()
    # cnn_model = Sequential()
    # # sigmoid part may need to be of form, activation='tanh'  activation = layers.sigmoid  input_shape=window_shape,
    # print(window_shape)
    # cnn_model.add(layers.ZeroPadding2D(padding=2, input_shape=window_shape, data_format='channels_last'))
    # # cnn_model.add(layers.ZeroPadding2D(padding=2, input_shape=(window_shape[2], window_shape[0], window_shape[1]), data_format='channels_first'))
    # cnn_model.add(layers.Conv2D(12, (5, 5), strides=(1, 1), activation ='relu'))
    # cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # cnn_model.add(layers.ZeroPadding2D(padding=(1, 1), data_format=None))
    # cnn_model.add(layers.Conv2D(4, (3, 3), strides=(1, 1), activation='relu'))
    # cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # cnn_model.add(layers.Flatten())
    # cnn_model.add(layers.Dense(20, activation='sigmoid'))  # layers.Dropout(rate, noise_shape=None, seed=None)
    # # cnn_model.add(layers.Dropout(0.1))
    # # cnn_model.add(layers.Flatten())
    # cnn_model.add(layers.Dense(2, activation='softmax'))  # layers.Dropout(rate, noise_shape=None, seed=None)
    # # cnn_model.add(layers.Dropout(0.1))
    # # cnn_model.add(layers.Activation('softmax'))
    #
    # ## Summary Line, Suppressed
    # # cnn_model.summary()
    #
    #
    # # cnn_model.add(layers.Dense())
    # cnn_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.load_weights('Keras_Models_Saved\Model_12')
    # cnn_model.load_weights('Keras_Models_Saved\Model_112917_2220_binary')
    # cnn_model.load_weights('Keras_Models_Saved\Model_112717_test7')
    # utils.plot_model(cnn_model, to_file='model_1107.png', show_shapes=True, show_layer_names=True, rankdir='LR')
    print(cnn_model.get_weights())
    return cnn_model


def retrieve_test_labels():
    labels = json.load(open('test_112917_2346.txt','r'))
    # labels = json.load(open('test_112717_test7.txt','r'))
    # print(labels[4])
    # cv2.waitKey(0)
    # labels = [labels[0]]
    return labels


if __name__ == '__main__':
    main()