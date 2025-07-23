import pandas as pd
import numpy as np

def load_and_preprocess_data():
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    # Clean and fill missing values
    for df in [df_train, df_test]:
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df.drop(columns=['Cabin'], inplace=True)
    df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
    df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)

    # Encode categorical
    for df in [df_train, df_test]:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df_train = pd.get_dummies(df_train, columns=['Embarked'])
    df_test = pd.get_dummies(df_test, columns=['Embarked'])

    # Align columns
    df_test = df_test.reindex(columns=df_train.columns.drop(['Survived']), fill_value=0)

    # Scale numeric
    for col in ['Age', 'Fare']:
        mean = df_train[col].mean()
        std = df_train[col].std()
        df_train[col] = (df_train[col] - mean) / std
        df_test[col] = (df_test[col] - mean) / std

    drop_cols = ['Name', 'Ticket']
    X_train = df_train.drop(columns=drop_cols + ['PassengerId', 'Survived']).values.astype(np.float64)
    y_train = df_train['Survived'].values.reshape(-1, 1).astype(np.float64)
    X_test = df_test.drop(columns=drop_cols + ['PassengerId']).values.astype(np.float64)
    passenger_ids = df_test['PassengerId']

    return X_train, y_train, X_test, passenger_ids
