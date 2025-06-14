import numpy as np 
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df_train = pd.read_csv("train.csv")
df_test  = pd.read_csv("test.csv")



df_train["cap-diameter"] = df_train["cap-diameter"].fillna(df_train["cap-diameter"].median())
df_train["stem-height"] = df_train["stem-height"].fillna(df_train["stem-height"].median())
df_train["stem-width"] = df_train["stem-width"].fillna(df_train["stem-width"].median())
X1 = df_train[["cap-diameter","stem-height","stem-width"]]


Categorical_df = df_train.drop(["cap-diameter","stem-height","stem-width"],axis=1)

y = Categorical_df["class"].map({"e":0,"p":1})

X2 = Categorical_df.drop(["class"],axis=1)

OE = OrdinalEncoder()

X2 = OE.fit_transform(X2)

X1 = X1.to_numpy()

X = np.stack([X1,X2],axis=1)

print(X.shape)
