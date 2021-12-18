from model import Model
from mninst_dataset import MnistDataset
import matplotlib.pyplot as plt


def test():
    # load image data
    data = MnistDataset()
    x_train = data.get_test_data()
    x_test = data.get_test_data(shuffle=True)

    # load or train model
    model = Model()
    if model.model_exists():
        model.load()
    else:
        model.build()
        model.train(x_train, epochs=200)
        model.save()

    # test autoencoder
    sample_size = min(len(x_test), 5)
    predictions = model.autoencode(x_test[:sample_size])

    fig = plt.figure()
    for i in range(sample_size):
        # original images
        fig.add_subplot(2, sample_size, i + 1)
        plt.imshow(x_test[i], cmap="gray")
        # decoded images
        fig.add_subplot(2, sample_size, i + sample_size + 1)
        plt.imshow(predictions[i], cmap="gray")

    plt.show()


test()
