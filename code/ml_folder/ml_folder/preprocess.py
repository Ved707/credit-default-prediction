import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'/home/vedant/Downloads/data_with_no_outliers/Clean_data.csv')

y = df['credit_card_default']
X = df.drop('credit_card_default', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022, stratify=y)


def preprocess(X):
    oec = OrdinalEncoder(categories=[
        ['Unknown', 'Low-skill Laborers', 'Laborers', 'Cleaning staff', 'Cooking staff', 'Security staff',
         'Waiters/barmen staff', 'Drivers', 'Private service staff',
         'Core staff', 'High skill tech staff', 'Sales staff', 'Accountants', 'Managers', 'Medicine staff', 'HR staff',
         'Secretaries', 'Realty agents', 'IT staff']])

    X[['occupation_type']] = oec.fit_transform(X[['occupation_type']])

    X = pd.get_dummies(X, drop_first=True)

    return X


scale = StandardScaler()
X_processed = preprocess(X_train)
X_scaled = scale.fit_transform(X_processed)

df1 = pd.DataFrame(X_scaled, index=X_train.index)
df2 = pd.DataFrame(y_train, index=df1.index)

train_df = pd.concat([df1, df2], axis=1)

train_df.to_csv(r'/home/vedant/Project/ml/data/train_data.csv')

X_test = preprocess(X_test)
X_test_scaled = scale.transform(X_test)

df1 = pd.DataFrame(X_test_scaled, index=X_test.index)
df2 = pd.DataFrame(y_test, index=df1.index)

test_df = pd.concat([df1, df2], axis=1)

test_df.to_csv(r'/home/vedant/Project/ml/data/test_data.csv')

