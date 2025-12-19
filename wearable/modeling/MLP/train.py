import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    DATASET_PATH,
    TARGET_ENC_COLUMN,
    FEATURES_ACC,
    TEST_PID,
    N_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE
)

from .preprocessing import build_preprocessor, apply_preprocessor
from .model import build_mlp, compile_model


def train_mlp():
    # Load dataset
    df = pd.read_parquet(DATASET_PATH)

    # Split train / test by participant
    df_train = df[df["pid"] != TEST_PID].reset_index(drop=True)

    X_train = df_train[FEATURES_ACC]
    y_train = df_train[TARGET_ENC_COLUMN]

    # Preprocessing
    preprocessor, feature_names = build_preprocessor(X_train)
    X_train_proc = apply_preprocessor(preprocessor, X_train, feature_names)

    X_train_tensor = X_train_proc.to_numpy().astype("float32")

    y_train_onehot = to_categorical(y_train, num_classes=N_CLASSES)

    # Class weights
    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights_values))

    # Model
    model = build_mlp(
        n_features=X_train_tensor.shape[1],
        n_classes=N_CLASSES
    )
    model = compile_model(model)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    # Training
    model.fit(
        X_train_tensor,
        y_train_onehot,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stop],
        shuffle=True,
        verbose=1
    )

    return model, preprocessor
