from model import *
from tensorflow import keras
import matplotlib.pyplot as plt


def test():
    # load data + normalize values
    data = keras.datasets.mnist
    (x_train, _), (x_test, _) = data.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = Model()
    if model.model_exists():
        model.load()
    else:
        model.build()
        model.train(x_train, epochs=5)
        model.save()

    # encoded image
    encoded = model.encode([x_test[0].reshape(-1, 28, 28, 1)])[0]

    # decoded image
    decoded = model.autoencode([x_test[0].reshape(-1, 28, 28, 1)])[0]

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 3, 1)
    plt.imshow(x_test[0], cmap="gray")
    fig.add_subplot(1, 3, 2)
    plt.imshow(encoded.reshape((8, 4)), cmap="gray")
    fig.add_subplot(1, 3, 3)
    plt.imshow(decoded, cmap="gray")
    plt.show()


test()
