from model import Model
from mninst_dataset import MnistDataset
import matplotlib.pyplot as plt


def test():
    test_path = "test"
    label = None

    # load image data
    data = MnistDataset()

    # load or train model
    model = Model()
    if model.model_exists(test_path):
        model.load(test_path)
    else:
        model.build()
        x_train = data.get_test_data(label)
        model.train(x_train, epochs=50)
        model.save(test_path)

    # test autoencoder
    x_test = data.get_test_data(label, shuffle=True)
    model.eval(x_test)

    sample_size = min(len(x_test), 10)
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
