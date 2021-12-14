from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
from tensorflow.keras import layers

from data import DataNew

input_shape = (1, 2048)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(2048, activation="softmax"),
    ]
)

model.summary()

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])

dataset = DataNew('midi_processed', 2048, 1)

model.fit_generator(dataset.generators_dict["train"], epochs=10)