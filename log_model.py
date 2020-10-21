import numpy as np
import pandas as pd 
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import pickle

if __name__ == "__main__":
    from os.path import join, dirname, abspath
    BASE_DIR = dirname(abspath("__file__"))
    DATA_DIR = join(BASE_DIR, "data")
    DATA_PATH = join(DATA_DIR, "clean_data.csv")
    MODEL_DIR = join(BASE_DIR, "models")

    df = pd.read_csv(DATA_PATH)
    X, y = get_X_y(df)

    # converting cat to int
    X_num = pd.get_dummies(X, drop_first=True)    
    
    # Over Sampling
    ros = RandomOverSampler(sampling_strategy=0.5)
    X_res, y_res = ros.fit_resample(X_num, y)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Logistic Regression
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # classification report
    class_report = classification_report(y_test, y_pred)
    print(class_report)

    # saving result
    with open(f"{BASE_DIR}/log_result.txt", "w") as results:
        results.write("Classification repost using Logistic Regression \n")
        for r in class_report:
            results.write(r)

    # saving model
    with open(f"{MODEL_DIR}/log_model.pkl", "wb") as pickle_out:
        pickle.dump(model, pickle_out)
    print("Model Saved")

    
