import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from consts import DIM


class Model:
    def __init__(self):
        self.encoder_path = "encoder/"
        self.decoder_path = "decoder/"
        self.autoencoder_path = "autoencoder/"

    def model_exists(self, path):
        print(Path(path) / self.autoencoder_path / "saved_model.pb")
        return (
            (Path(path) / self.autoencoder_path / "saved_model.pb").exists()
            and (Path(path) / self.encoder_path / "saved_model.pb").exists()
            and (Path(path) / self.decoder_path / "saved_model.pb").exists()
        )

    def build(self, lr=0.001):
        # encoder
        self.encoder = keras.Sequential(
            [
                layers.Flatten(input_shape=[DIM, DIM, 1]),
                layers.Dense(128, activation="leaky_relu"),
                layers.Dense(64, activation="leaky_relu"),
                layers.Dense(32, activation="sigmoid"),
            ],
            name="encoder",
        )

        # decoder
        self.decoder = keras.Sequential(
            [
                layers.Dense(64, activation="leaky_relu", input_shape=[32]),
                layers.Dense(128, activation="leaky_relu"),
                layers.Dense(DIM * DIM, activation="sigmoid"),
                layers.Reshape([28, 28, 1]),
            ],
            name="decoder",
        )

        # combine to autoencoder
        self.autoencoder = keras.Sequential(
            [self.encoder, self.decoder], name="autoencoder"
        )

        self.autoencoder.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=["accuracy"],
        )

        print(self.autoencoder.summary())
        print(self.encoder.summary())
        print(self.decoder.summary())

    def train(self, x_train, epochs=10, batch_size=32, shuffle=True):
        self.autoencoder.fit(
            x_train, x_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle
        )

    def eval(self, x_test, batch_size=32):
        self.autoencoder.evaluate(x_test, x_test, batch_size)

    def save(self, path):
        self.encoder.save(Path(path) / self.encoder_path)
        self.decoder.save(Path(path) / self.decoder_path)
        self.autoencoder.save(Path(path) / self.autoencoder_path)

    def load(self, path):
        self.encoder = keras.models.load_model(Path(path) / self.encoder_path)
        self.decoder = keras.models.load_model(Path(path) / self.decoder_path)
        self.autoencoder = keras.models.load_model(Path(path) / self.autoencoder_path)

    def encode(self, data):
        return self.encoder.predict(data)

    def decode(self, data):
        return self.decoder.predict(data)

    def autoencode(self, data):
        return self.autoencoder.predict(data)

    def encoder_dims(self):
        return self.encoder.input_shape, self.encoder.output_shape

    def decoder_dims(self):
        return self.decoder.input_shape, self.decoder.output_shape
