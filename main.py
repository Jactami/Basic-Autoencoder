import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from model import Model
from mninst_dataset import MnistDataset
import random
import tensorflow as tf
import shutil


def create_model(label, epochs, path):
    # get train data
    data = MnistDataset()
    if label:
        x_train = data.get_train_data(label)
    else:
        x_train = data.get_train_data()

    # train model
    model = Model()
    model.build()
    model.train(x_train, epochs)
    model.save(path)

    return model


def generate_images(model, n):
    # generate encoded input data
    (_, dim), _ = model.decoder_dims()
    x_input = []
    for _ in range(n):
        x_input.append([random.random() for _ in range(dim)])

    x_input = tf.convert_to_tensor(x_input)

    # generate decoded images
    x_output = model.decode(x_input)

    return x_output


def save_images(images, path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    nf = len(str(len(images)))
    for i in range(len(images)):
        name = path + "img" + str(i).zfill(nf) + ".png"
        tf.keras.utils.save_img(name, images[i])

    print(f"images saved in '{path}'")


if __name__ == "__main__":
    # specify number of generated images
    while True:
        try:
            n = int(input("Specify the number of images: "))
            if n < 0:
                raise ValueError
            else:
                break
        except ValueError:
            print("Error! Amount of images must be a positive number.")

    # retrieve decoder
    model_path = "models"
    model = Model()
    choice = "n"

    # check for pre-trained models
    if model.model_exists(model_path):
        choice = ""
        valid_choices = ["y", "n"]
        while not choice in valid_choices:
            choice = input("Existing model found. Use this model? (y/n): ")
            choice = choice.lower()

    if choice == "y":
        model.load(model_path)
    elif choice == "n":
        # no existing model found or existing model should not be used: make new model
        labels = MnistDataset.labels
        while True:
            try:
                label = input(
                    f"(Optional) Specify a label from {labels}. Skip otherwise: "
                )
                if label and not label in labels:
                    raise ValueError
                else:
                    break
            except ValueError:
                print("Error! Input must be a valid label or empty.")

        # specify training epochs
        while True:
            try:
                epochs = int(input("Specify the number of training epochs: "))
                if epochs < 0:
                    raise ValueError
                else:
                    break
            except ValueError:
                print("Error! Epochs must be a positive number.")

        if label:
            model = create_model(label, epochs, model_path)
        else:
            model = create_model(None, epochs, model_path)

    # generate predicted image data
    images = generate_images(model, n)

    # save images
    save_images(images, "output/")
