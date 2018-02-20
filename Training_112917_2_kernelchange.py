import os
import time
import numpy as np
import cv2
import sklearn.model_selection as model
import xml.etree.ElementTree as ET
import pandas as pd
import skimage.transform as transform
print('Starting, preparing Keras . . . ' + time.asctime(time.localtime()))
from keras.models import Sequential
from keras import optimizers
from keras import regularizers
import keras.layers as layers
from keras import backend as K
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def main():
    time.clock()
    current_dir = directory()
    training_label_dir = directory('training label directory')
    image_dir = directory('training image directory')
    append = input('Please type the appended string to add to save data: ')
    training_label_names = list_file_names(training_label_dir)
    training_image_names = list_file_names(image_dir)
    training(current_dir, training_label_dir, image_dir, training_label_names, training_image_names, append)
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
    return directory1


def list_file_names(dir):
    f = []
    filenames = next(os.walk(dir))[2]
    f.extend(filenames)
    return f


def training(dir_c, dir_l, dir_i, labels, images, append):
    train_labels, test_labels = generate_train_test(labels, append)
    window_shape = (64, 64, 3)
    print('Prepare Model . . . ' + time.asctime(time.localtime()))
    cnn_model = create_CNN(window_shape)
    print('Begin Loop . . . ' + time.asctime(time.localtime()))

    train_loop(train_labels, dir_c, dir_i, dir_l, cnn_model, window_shape, append)
    return None


def generate_train_test(labels, append):
    train_labels, test_labels = model.train_test_split(labels, test_size = 0.250, shuffle = True)
    f = open('test_' + append + '.txt', 'w')
    json.dump(test_labels, f)
    f.close()
    f = open('train_' + append + '.txt', 'w')
    json.dump(train_labels, f)
    f.close()
    return train_labels, test_labels


def train_loop(labels, dir_c, dir_i, dir_l, cnn_model, window_shape, append):
    class_weights = {0: 1., 1: 50.}
    area_window = window_shape[0] * window_shape[1]
    total_images = len(labels)
    total_iteration = 1
    for label in labels:
        image_data = retrieve_labels(label, dir_l)
        # print(image_data)
        number_of_plates = image_data.shape[0]
        boxes_percent = np.empty((number_of_plates, 4))
        for i in range(0, number_of_plates):
            boxes_percent[i][0] = image_data.at[i, 'y_Percent'][0]
            boxes_percent[i][1] = image_data.at[i, 'y_Percent'][1]
            boxes_percent[i][2] = image_data.at[i, 'x_Percent'][0]
            boxes_percent[i][3] = image_data.at[i, 'x_Percent'][1]
        image, name = load_image(label, dir_i)
        pyr = tuple(transform.pyramid_gaussian(image, downscale=2, max_layer=3))
        shift_number = (24, 12, 8, 4)  #, 2)
        epochs = (400, 180, 60, 20)
        iteration = 3
        for p in pyr[iteration:]:  # change back to 0:
            bndbox = find_bndbox(p, boxes_percent)
            epochs_use = epochs[iteration]
            gen = sliding_window(p, shift_number[iteration], window_shape, bndbox, number_of_plates)
            print('Training - Image: ' + str(total_iteration) + '/' + str(total_images) + ' Size: '
                + str(iteration+1) + '/4    ' + time.asctime(time.localtime()))
            cnn_model.fit_generator(gen, 84, epochs=epochs_use, class_weight=class_weights, verbose=2)
            iteration += 1
        total_iteration += 1
    cnn_model.save_weights(dir_c + "\Keras_Models\Model_" + append)
    print('DONE!!!   ' + time.asctime(time.localtime()))


def load_image(label, dir):
    label = label.rstrip('.xml') + '.jpg'
    image_path = dir + '//' +label
    image = cv2.imread(image_path)
    return image, label


def retrieve_labels(label, dir):
    dir = dir + '//' + label
    tree = ET.parse(dir)
    root = tree.getroot()
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    c6 = []
    for data in root.findall('object'):
        plate_chars = data.find('name').text
        c1.append(plate_chars)
        bndbox=data.find('bndbox')
        box_x = (int(bndbox.find('xmin').text), int(bndbox.find('xmax').text))
        box_y = (int(bndbox.find('ymin').text), int(bndbox.find('ymax').text))
        c2.append(box_x)
        c3.append(box_y)
    image_size = root.find('size')
    image_size_1 = (int(image_size.find('width').text), int(image_size.find('height').text))
    c4 = [image_size_1]*len(c1)
    iter = 0
    for item in c2:
        c5.append((float(item[0]) / image_size_1[0], float(item[1]) / image_size_1[0]))
        c6.append((float(c3[iter][0]) / image_size_1[1], float(c3[iter][1]) / image_size_1[1]))
        iter = iter + 1
    image_data = pd.DataFrame({'Plate' : c1, 'Bounding_Box_x' : c2, 'Bounding_Box_y' : c3, 'Image Size' : c4, 'x_Percent': c5, 'y_Percent' : c6})
    return image_data


def sliding_window(image, shift_number, window_shape, bndbox, number_of_plates):
    rows, cols, dim = image.shape
    area = rows * cols
    window_shape = window_shape
    while True:
        for r in range(0, rows - window_shape[0], shift_number):
            for c in range(0, cols- window_shape[1], shift_number):
                bounds = (r, r+window_shape[0], c, c+window_shape[1])
                new_image = image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
                output = plate_presence_labeler(area, bounds, bndbox, number_of_plates)
                new_image = new_image[np.newaxis, ...]
                # plate_shower(new_image, output)
                yield (new_image, output)
            bounds = (r, r+window_shape[0], cols-window_shape[1], cols)
            new_image = image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            output = plate_presence_labeler(area, bounds, bndbox, number_of_plates)
            new_image = new_image[np.newaxis, ...]
            # plate_shower(new_image, output)
            yield (new_image, output)
        bounds = (rows-window_shape[0], rows, 0, window_shape[1])
        for c in range(0, cols - window_shape[1], shift_number):
            bounds = (rows-window_shape[0], rows, c, c + window_shape[1])
            new_image = image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            output = plate_presence_labeler(area, bounds, bndbox, number_of_plates)
            new_image = new_image[np.newaxis, ...]
            # plate_shower(new_image, output)
            yield (new_image, output)
        bounds =(rows-window_shape[0], rows, cols - window_shape[1], cols)
        new_image = image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
        output = plate_presence_labeler(area, bounds, bndbox,number_of_plates)
        new_image = new_image[np.newaxis, ...]
        # plate_shower(new_image, output)
        yield (new_image, output)


def plate_shower(new_image, output):
    if output[0][0] == 1:
        cv2.imshow('positive', new_image[0])
        cv2.waitKey(0)
        cv2.destroyWindow('positive')
    return None


def plate_presence_labeler(area_window, location, bndbox, number_of_plates):
    y = np.asarray(0)
    y = y[np.newaxis, ...]
    # print(location)
    # print(bndbox)
    for i in range(0, number_of_plates):
        if location[0] <= bndbox[i][0] and location[1] >= bndbox[i][1] and location[2] <= bndbox[i][2] and location[3] >= bndbox[i][3]:
            # print('box in window')
            area_bndbox = (bndbox[i][1] - bndbox[i][0]) * (bndbox[i][3] - bndbox[i][2])
            # print('bndbox area', area_bndbox)
            # print('bndbox window', area_window)
            if area_bndbox >= 110:  # 0.025 * area_window:
                print('plate large enough')
                y = np.asarray(1)
                y = y[np.newaxis, ...]
    return y


def create_CNN(window_shape):
    K.set_image_dim_ordering('tf')
    # sgd = optimizers.SGD(lr=0.0000001)  #, decay = 0.00001)
    adam = optimizers.Adam(lr=0.00000005)  #, decay=0.00005)
    cnn_model = Sequential()
    print(window_shape)
    cnn_model.add(layers.ZeroPadding2D(padding=2, input_shape=window_shape, data_format='channels_last'))
    cnn_model.add(layers.Conv2D(12, (5, 5), strides=(1, 1), activation='relu'))  #, activity_regularizer=regularizers.l1()))  #'relu'))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # cnn_model.add(layers.Dropout(0.5))
    cnn_model.add(layers.ZeroPadding2D(padding=(1, 1), data_format=None))
    cnn_model.add(layers.Conv2D(24, (3, 3), strides=(1, 1), activation='relu'))  #, activity_regularizer=regularizers.l1()))  #'relu')) #4 previously
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # cnn_model.add(layers.Dropout(0.5))
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(20, activation='relu'))  #, activity_regularizer=regularizers.l2()))  # 'sigmoid'))  # layers.Dropout(rate, noise_shape=None, seed=None)
    cnn_model.add(layers.Dropout(0.25))
    # cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(1, activation='sigmoid'))  #, kernel_regularizer=regularizers.l2()))  # layers.Dropout(rate, noise_shape=None, seed=None)
    # cnn_model.add(layers.Dropout(0.5))
    # cnn_model.add(layers.Activation('softmax'))
    cnn_model.summary()
    # cnn_model.add(layers.Dense())
    cnn_model.compile(optimizer=adam, loss='binary_crossentropy')
    return cnn_model


def find_bndbox(image, boxes_percent):
    size = image.shape[0], image.shape[0], image.shape[1], image.shape[1]
    size = np.array(size)
    bndbox = np.multiply(boxes_percent, size)
    bndbox = bndbox.astype(np.int64)
    # for i in range(0, bndbox.shape[0]):
    #     cv2.rectangle(image, (bndbox[i][3], bndbox[i][1]), (bndbox[i][2], bndbox[i][0]), (0, 0, 255), 3)
    # print(size)
    # print(bndbox)
    # cv2.imshow('plates', image)
    # cv2.waitKey(0)
    # cv2.destroyWindow('plates')
    return bndbox


if __name__ == '__main__':
    main()