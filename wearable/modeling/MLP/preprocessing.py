import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from .config import (
    CATEGORICAL_FEATURES,
    PASSTHROUGH_FEATURES,
    FILLNA_ZERO_COLUMNS
)


def build_preprocessor(X_train):
    """
    Builds and fits a ColumnTransformer for preprocessing.

    - Numeric features: StandardScaler
    - Categorical features: OneHotEncoder
    - Passthrough features: unchanged
    """

    # Fill NA columns if needed
    X_train = X_train.copy()
    for col in FILLNA_ZERO_COLUMNS:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(0)

    numeric_features = [
        col for col in X_train.columns
        if col not in CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
    ]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("pass", "passthrough", PASSTHROUGH_FEATURES),
        ],
        remainder="drop"
    )

    preprocessor.fit(X_train)

    feature_names = (
        numeric_features +
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES).tolist() +
        PASSTHROUGH_FEATURES
    )

    return preprocessor, feature_names


def apply_preprocessor(preprocessor, X, feature_names):
    X_trans = preprocessor.transform(X)
    return pd.DataFrame(X_trans, columns=feature_names, index=X.index)
