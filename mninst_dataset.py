from keras.datasets import mnist
import tensorflow as tf


class MnistDataset:
    labels = [str(x) for x in range(10)]

    def __init__(self):
        (train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()
        self.train_data = self._process_data(train_imgs, train_labels)
        self.test_data = self._process_data(test_imgs, test_labels)

    def get_train_data(self, label=None, shuffle=False):
        return self._get_data(self.train_data, label, shuffle)

    def get_test_data(self, label=None, shuffle=False):
        return self._get_data(self.test_data, label, shuffle)

    def _get_data(self, dictionary, label, shuffle):
        data = tf.zeros([0, 28, 28], dtype=tf.dtypes.float64)

        if label:
            if self._is_valid_label(label):
                data = dictionary[label]
        else:  # no label given: return entire dataset
            for label in MnistDataset.labels:
                data = tf.concat([data, dictionary[label]], 0)

        if shuffle:
            data = tf.random.shuffle(data)

        return data

    def _process_data(self, img_data, img_labels):
        # normalize pixel values
        img_data = img_data / 255.0

        # build dictionary
        dictionary = {}
        for label in MnistDataset.labels:
            mask = [str(x) == label for x in img_labels]
            imgs = tf.boolean_mask(img_data, mask)
            dictionary[label] = imgs

        return dictionary

    def _is_valid_label(self, label):
        return label in MnistDataset.labels
