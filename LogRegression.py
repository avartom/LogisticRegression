from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

def wrangle(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df.dropna(subset=["overall"], inplace=True)
    df["Great"] = (df["overall"] >=4).astype(int)
    df = df.drop(columns=['Notes', 'Location', 'Address', 'URL', 'Neighborhood'])
    df = df.drop(columns=['Rec', 'overall'])
    return df
df = wrangle("/Users/ara_vartomian/Downloads/Burrito - 10D.csv")

def split_data(df):
    target = "Great"
    X = df.drop(columns=target)
    y = df[target]
    cutoff = '2018'
    mask = X.index < cutoff
    X_train, y_train = X.loc[mask], y.loc[mask]
    X_test, y_test = X[~mask], y[~mask]
    return (X_train, X_test, y_train, y_test, y.value_counts(normalize=True))
sp_data = split_data(df)

def model (X_train, X_test, y_train, y_test):
    model_logr = make_pipeline(OneHotEncoder(),
                           SimpleImputer(), StandardScaler(),
                           LogisticRegression())

    model_logr.fit(X_train, y_train)
    training_acc = model_logr.score(X_train, y_train)
    test_acc = model_logr.score(X_test, y_test)
    return (training_acc, test_acc)
mod = model(sp_data[0], sp_data[1], sp_data[2], sp_data[3])
print(mod[0], mod[1])


