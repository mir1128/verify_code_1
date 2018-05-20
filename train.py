from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import numpy as np
import glob
import os
from keras.preprocessing import image
from model import model


def load_data(dir_name, test_set_size):
    captcha_image_files = glob.glob(os.path.join(dir_name, "*"))

    data = []
    labels = []

    for (i, captcha_image_file) in enumerate(captcha_image_files):
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]
        img = image.load_img(captcha_image_file, target_size=(20, 20))
        x = image.img_to_array(img)

        data.append(x)
        labels.append(captcha_correct_text.split("_")[0])

    data = np.array(data, dtype="float") / 255.0

    label_list = list('23456789ABCDEFGHJKLMNPQRSTUVWXYZ')
    lb = LabelBinarizer()
    trans = lb.fit(label_list)
    labels_bin = np.array(trans.transform(labels))

    return train_test_split(data, labels_bin, test_size=test_set_size, random_state=0)


if __name__ == '__main__':
    (X_train, X_test, Y_train, Y_test) = load_data("training_images", 0.25)
    print (X_train.shape)
    print (X_test.shape)
    print (Y_train.shape)
    print (Y_test.shape)
    # model(X_train, Y_train, X_test, Y_test)

    start = 0
    end = 10

    model(X_train, Y_train, None, None, operation='restore', predict=X_test[start:end, :, :, :])
    print "---------------------------------"
    print (np.argmax(Y_test[start:end, :], 1))

