"""
Michael Novick 2017
Python License Plate Detection - Training -  Synthesized Images
This script trains a Convolutional Neural Network (CNN) to detect license plates in images. It can be easily modified
for other forms of detection. Training images are synthesized and come in three varieties: background with
superimposed plate, blank background without plate, and blank background with superimposed background. The superimposed
background category was included to prevent the network from learning that sharp transitions indicated plate presence,
because images with real plates are not likely to have the sharp transitions that image superposition introduces.
"""

# Import necessary packages. Versions of each package are listed on Github at URL:
# https://github.com/Michael-Novick/License-Plate-Detection/blob/master/README.md
#
# They were all installed manually using Pip install. Several problems arose from using Conda install in conjunction
# with Pip install.
# For clarity, I have included them below:
import os                                   # Python 3.6.2 (included)
import time                                 # Python 3.6.2 (included)
import numpy as np                          # 1.13.3+mkl
import cv2                                  # opencv-python 3.3.0
import sklearn.model_selection as model     # skikit-learn 0.19.1
from keras.models import Sequential         # 2.0.8
from keras import optimizers                # 2.0.8
from keras import regularizers              # 2.0.8
import keras.layers as layers               # 2.0.8
from keras import backend as K              # 2.0.8
import random                               # Python 3.6.2 (included)
import imutils                              # 0.4.3
# TensorFlow is needed for Keras            # 1.4.0


def main():
    # Start clock.
    time.clock()

    # Find directory of script.
    current_dir = directory()

    # This call asks for a string to append for files associated with the run training session.
    append = input('Please type the appended string to add to save data: ')

    # This establishes the directory of the plate images and background images, and are hardcoded locations to folders
    # of images on my machine.
    dir_p = current_dir+'\Plates_122617'
    dir_bg = current_dir+'\Backgrounds_122617'

    # This calls custom method for training. It was established before the script architecture was clear, and it is now
    # clear that this separate method doesn't provide any particular gains over writing code in the main() method.
    train_loop(current_dir, append, dir_p, dir_bg)
    return None


def directory(prompt=None):
    """
    This method finds paths for subdirectories from the location of the file. Two are hardcoded specifically for my
    machine. It is a bit clumsy.
    :param prompt: indicator for the user to type the relative path from the script folder to the prompted folder.
    :return: String of desired path.
    """
    # Creates string of absolute path to the script.
    directory1 = os.path.dirname(os.path.abspath(__file__))

    # Two hardcoded directories.
    if prompt == 'training label directory':
        directory1 = directory1 + '\Training_MyPictures_101517_Labels'
    elif prompt == 'training image directory':
        directory1 = directory1 + '\Training_MyPictures_101517'
    elif prompt is None:
        None
    else:
        # user imputs the remainder of the desired path.
        directory1 = directory1 + input('Please indicate: ' + prompt + ': ')
    return directory1


def list_file_names(dir):
    """
    This method returns a list of file names in a specific directory, and omits directories in the directory. In other
    words, if given a path to an image folder, it will return a list of image names in that folder, but will omit names
    of other subfolders in that folder and will not list the contents of any subfolders.
    :param dir: String of absolute path to the folder of interest.
    :return: List of just the files (no folders) in the folder at the inputted directory.
    """
    # Initialize list.
    f = []

    # Identify just file names in directory and append to list f.
    filenames = next(os.walk(dir))[2]
    f.extend(filenames)
    return f


def train_loop(dir_c, append, dir_p, dir_bg):
    """
    This method runs the training of the neural network and was separated from the main() method when training and
    testing was going to be conducted in one script. This is no longer the case.
    :param dir_c: Path string to directory of script.
    :param append: Informative string to append to training session-specific files.
    :param dir_p: Path string to directory of plates.
    :param dir_bg: Path string to directory of backgrounds.
    :return: None
    """
    # Shape of input to neural network.
    window_shape = (64, 64, 3)

    # Statement used to gain insight on time taken before training starts.
    print('Prepare Model . . . ' + time.asctime(time.localtime()))

    # Call to method that creates and returns a neural network model.
    cnn_model = create_CNN(window_shape)

    # Statement used to gauge training time.
    print('Begin Loop . . . ' + time.asctime(time.localtime()))

    # Class weights for inputs, to counter class imbalance during training.
    class_weights = {0: 1., 1: 2.}

    # Names of plates and backgrounds in list form from the plate directory and background directory.
    plate_names = list_file_names(dir_p)
    bg_names = list_file_names(dir_bg)

    # Call method that produces the generator for continuous image synthesizing.
    gen = image_synthesizer(dir_p, dir_bg, plate_names, bg_names)

    # Call for training the neural network model for specified epochs and batch size. Also saves model weights using
    # the tag from the beginning of the script (for identification) so the model can be tested.
    cnn_model.fit_generator(gen, 32, epochs = 120, class_weight=class_weights, verbose=1)
    cnn_model.save_weights(dir_c + "\Keras_Models\Model_" + append)

    # Indicate completed testing.
    print('DONE!!!   ' + time.asctime(time.localtime()))
    return None


def create_CNN(window_shape):
    """
    This method creates and returns the specified neural network architecture. There are several hyperparameters to
    modify.
    :param window_shape: Shape of input window.
    :return: Return Keras model of neural network.
    """

    # Indicates that TensorFlow is the backend being used.
    K.set_image_dim_ordering('tf')

    # Potential optimizers. Learning rate (lr) and decay can be modified as necessary.
    # sgd = optimizers.SGD(lr=0.0000001)  #, decay = 0.00001)
    adam = optimizers.Adam(lr=5*10**(-3), decay=4*10**(-6))

    # Initialize sequential neural network model
    cnn_model = Sequential()

    # Add layers. Zero padding to allow full-sized convolution output. There are countless architectures to choose; this
    # was the architecture I experimented with. Each layer has sseveral input parameters to tune, which is where I spent
    # probably 80%-90% of my time.
    cnn_model.add(layers.ZeroPadding2D(padding=2, input_shape=window_shape, data_format='channels_last'))
    cnn_model.add(layers.Conv2D(18, (5, 5), strides=(1, 1), activation='relu',
                                activity_regularizer=regularizers.l2(0.000000001), bias_regularizer=regularizers.l2(0.000000001)))  #'relu'))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # cnn_model.add(layers.Dropout(0.25))
    cnn_model.add(layers.ZeroPadding2D(padding=(1, 1), data_format=None))
    cnn_model.add(layers.Conv2D(36, (3, 3), strides=(1, 1), activation='relu',
                                activity_regularizer=regularizers.l2(0.000000001), bias_regularizer=regularizers.l2(0.000000001)))  #'relu')) #4 previously
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # cnn_model.add(layers.Dropout(0.25))
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(20, activation='relu',
                               activity_regularizer=regularizers.l2(0.000000001), bias_regularizer=regularizers.l2(0.000000001)))  # 'sigmoid'))  # layers.Dropout(rate, noise_shape=None, seed=None)
    # cnn_model.add(layers.Dropout(0.25))
    # cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(1, activation='sigmoid',
                               activity_regularizer=regularizers.l2(0.000000001), bias_regularizer=regularizers.l2(0.000000001)))  # layers.Dropout(rate, noise_shape=None, seed=None)
    # cnn_model.add(layers.Dropout(0.5))
    # cnn_model.add(layers.Activation('softmax'))

    # This line prints a model summary
    cnn_model.summary()
    # cnn_model.add(layers.Dense())

    # This line compiles the model, and returns it.
    cnn_model.compile(optimizer=adam, loss='binary_crossentropy')
    return cnn_model


def image_synthesizer(dir_p, dir_bg, platenames, bgnames):
    """
    This method creates and returns a random image synthesizer using directories of plates and backgrounds.
    :param dir_p: plate image directory
    :param dir_bg: background image directory
    :param platenames: list of directory contents
    :param bgnames: list of directory contents
    :return: synthesized plate image
    """
    while True:  # needed for generator method for continuous recall
        # load a random background image
        bg_r = cv2.imread(dir_bg + '\\' + random.choice(bgnames))

        # randomly scale for background
        rand_bg_1 = np.random.randint(int(0.0625 * bg_r.shape[1]), int(
            0.9375 * bg_r.shape[1]))  ## Some images have dims of 720, too small for 0.0625. BEWARE OF POTENTIAL ERROR
        bg_r = imutils.resize(bg_r, width=rand_bg_1)

        # take random snip of scaled background with a size corresponding to input window
        rand_bg_2 = np.random.randint(0, bg_r.shape[0] - 64)
        rand_bg_3 = np.random.randint(0, bg_r.shape[1] - 64)
        bg_r2 = bg_r[rand_bg_2:rand_bg_2 + 64, rand_bg_3:rand_bg_3 + 64]

        # this statement prepares the output for a "no plate" scenario. Read below, where the other two scenarios are
        # included
        bg_r3 = bg_r2.copy()
        flag = 0

        # randomly generate 1 of 3 numbers to select one of three scenarios: just a background without plate, a
        # background with a superimposed background and no plate, or a background with superimposed plate. The
        # background and background was included to eliminate the chance the network would learn to recognize harsh
        # edges due to image superposition. This way superpositioned edges would just as likely indicate a plate as it
        # would indicate no plate (1/3 chance for either)
        z = np.random.randint(0, 3)
        if z == 0:   # plate on background, output (flag) should be 1
            plate_r = cv2.imread(dir_p + '\\' + random.choice(platenames))
            rand = np.random.randint(28, 65)
            plate_r = imutils.resize(plate_r, width=rand)
            # print(plate_r.shape)
            # cv2.imshow('plate_r', plate_r)
            # cv2.waitKey(0)
            y = np.random.randint(0, 65 - plate_r.shape[0])
            x = np.random.randint(0, 65 - rand)
            bg_r3[y:y + plate_r.shape[0], x:x + rand] = plate_r
            flag = 1

        elif z == 1:  # background on background, output (flag) should be 0
            fakeplate_r = cv2.imread(dir_bg + '\\' + random.choice(bgnames))
            rand_fakeplate_1 = np.random.randint(int(0.0625 * fakeplate_r.shape[1]), int(
                0.9375 * fakeplate_r.shape[1]))  ## Some images have dims of 720, too small for 0.0625. beware
            fakeplate_r = imutils.resize(fakeplate_r, width=rand_fakeplate_1)
            rand_fakeplate_2 = np.random.randint(0, fakeplate_r.shape[0] - 64)
            rand_fakeplate_3 = np.random.randint(0, fakeplate_r.shape[1] - 64)
            rand_1 = np.random.randint(32, 65)
            rand_2 = np.random.randint(24, 54)
            fakeplate_r2 = fakeplate_r[rand_fakeplate_2:rand_fakeplate_2 + rand_2,
                           rand_fakeplate_3:rand_fakeplate_3 + rand_1]
            # print(plate_r.shape)
            # cv2.imshow('plate_r', plate_r)
            # cv2.waitKey(0)
            y = np.random.randint(0, 65 - fakeplate_r2.shape[0])
            x = np.random.randint(0, 65 - fakeplate_r2.shape[1])
            bg_r3[y:y + fakeplate_r2.shape[0], x:x + fakeplate_r2.shape[1]] = fakeplate_r2

        # Add additional dimensions to the output for preparation for input to model, and return the output.
        bg_r3 = bg_r3[np.newaxis, ...]
        flag = np.asarray(flag)
        flag = flag[np.newaxis, ...]
        yield (bg_r3, flag)
        # cv2.imshow('bg_r', bg_r3)
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        # print(iter, z, flag)
        # iter += 1


# Statement used to run the main() method.
if __name__ == '__main__':
    main()