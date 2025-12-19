import pandas as pd
import joblib

from . import config
from . import random_forest as RF


def train_random_forest(save_model: bool = True):
    """
    Trains the Random Forest model using the prepared Capture-24 dataset.
    Returns the trained model.
    """

    # Load dataset
    df = pd.read_parquet(config.DATAFILE)

    # Feature preparation
    df = RF.compress(df)
    X = RF.prepare_X(df)
    y, target_col = RF.select_target(df)

    # Train / test split
    X_train, X_test, y_train, y_test = RF.split_data(X, y)

    # Coordinate adjustment & imputation
    X_train, X_test = RF.adjust_coor(X_train, X_test)
    X_train_imp, X_test_imp = RF.impute_missing(X_train, X_test)

    # Handle imbalance
    X_train_res, y_train_res = RF.apply_smote(X_train_imp, y_train)

    # Train model
    model = RF.train_model(X_train_res, y_train_res)

    # Evaluation (console output only)
    RF.evaluate_model(model, X_test_imp, y_test, target_col)

    # Persist model
    if save_model:
        joblib.dump(model, config.MODEL_OUTPUT_PATH)
        print(f"\n[OK] Random Forest model saved at {config.MODEL_OUTPUT_PATH}")

    return model
