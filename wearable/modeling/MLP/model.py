from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from .config import (
    MLP_HIDDEN_LAYERS,
    DROPOUT_RATES,
    LEARNING_RATE
)

def build_mlp(n_features, n_classes):
    model = models.Sequential()

    for i, (units, dropout) in enumerate(
        zip(MLP_HIDDEN_LAYERS, DROPOUT_RATES)
    ):
        if i == 0:
            model.add(layers.Dense(units, input_shape=(n_features,), activation=None))
        else:
            model.add(layers.Dense(units, activation=None))

        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(n_classes, activation="softmax"))

    return model


def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
