import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import config

def compress(df, **kwargs):
    """
    Reduces the size of the DataFrame by downcasting numerical columns
    """
    input_size = df.memory_usage(index=True).sum()/ 1024**2  #retorna qnto de memoria cada coluna ocupa || soma
                                                             # os valores, incluindo o indice || converte para megabytes
                                                             # 1kb = 1024 bytes => 1mb = 1024**2 bytes
    print("old dataframe size: ", round(input_size,2), 'MB')

    in_size = df.memory_usage(index=True).sum()

    for t in ["float", "integer"]:                           # percorre os floats e os integers e pra cada um cria
                                                             # uma lista l_cols com as colunas do df q tem esse tipo de dado
        l_cols = list(df.select_dtypes(include=t))

        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=t)

    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100

    print("optimized size by {} %".format(round(ratio,2)))
    print("new DataFrame size: ", round(out_size / 1024**2,2), " MB")

    return df

def select_target(df, label_choice=config.LABEL_CHOICE):
    """Seleciona a label target a ser usada pelo modelo (1: WillettsMET2018, 2: WillettsSpecific2018 ou 3:Walmsley2020 )"""
    print(f"[select_target] Iniciando seleção da label target ({label_choice})...")

    target_col = config.TARGET_LABELS[label_choice]
    print(f"[select_target] Target selecionada: {target_col}")
    y = df[target_col]
    print("[select_target] Finalizado!\n")
    return y, target_col


def prepare_X(df: pd.DataFrame):
    X = df.drop(columns=config.COLS_TO_DROP)
    print("[prepare_X] Finalizado!\n")
    return X


def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def adjust_coor(X_train, X_test):
    cols_corr = ['corr_xy', 'corr_xz', 'corr_yz']
    X_train[cols_corr] = X_train[cols_corr].fillna(0)
    X_test[cols_corr]  = X_test[cols_corr].fillna(0)
    return X_train, X_test

def impute_missing(X_train, X_test):

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)
    print("Imputed missing\n")
    return X_train_imp, X_test_imp


def apply_smote(X_train_imp, y_train):
    sm = SMOTE(
        k_neighbors=config.SMOTE_K,
        random_state=config.RANDOM_STATE,
        sampling_strategy=config.SMOTE_STRATEGY
        )
    X_train_res, y_train_res = sm.fit_resample(X_train_imp, y_train)

    return X_train_res, y_train_res

def train_model(X_train_res, y_train_res):
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        class_weight=config.RF_CLASS_WEIGHT,
        n_jobs=config.RF_JOBS,
        random_state=config.RANDOM_STATE
    )
    rf.fit(X_train_res, y_train_res)
    return rf

def evaluate_model(model, X_test_imp, y_test, target_col):
    y_pred = model.predict(X_test_imp)
    print(f"** CLASSIFICATION REPORT: {target_col} **")
    print(classification_report(y_test, y_pred))
