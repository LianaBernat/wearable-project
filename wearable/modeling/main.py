import pandas as pd
import config
import random_forest as RF
import joblib


df = pd.read_parquet(config.DATAFILE)
df = RF.compress(df)

X = RF.prepare_X(df)

y, target_col = RF.select_target(df)

X_train, X_test, y_train, y_test = RF.split_data(X, y)
X_train, X_test = RF.adjust_coor(X_train, X_test)
X_train_imp, X_test_imp = RF.impute_missing(X_train, X_test)

X_train_res, y_train_res = RF.apply_smote(X_train_imp, y_train)

model = RF.train_model(X_train_res, y_train_res)

RF.evaluate_model(model, X_test_imp, y_test, target_col)

joblib.dump(model, "randomforest.joblib")
print("\nModelo salvo em randomforest.joblib")

#joblib.dump(imputer, "../model_rf/imputer.joblib")
#print("Imputer salvo em model_rf/imputer.joblib")
