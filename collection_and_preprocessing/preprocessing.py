import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class GasPricePredictor:
    def __init__(self, data_path, random_state=42, n_estimators=1000):
        self.data_path = data_path
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(random_state=self.random_state, n_estimators=self.n_estimators)

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        df['label'] = df['close'].diff().gt(0).astype(int).shift(-1).fillna(0)
        return df

    def split_data(self, df):
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=self.random_state)
        X_train = train_df.drop(['label', 'timestamp'], axis=1)
        y_train = train_df['label']
        X_test = test_df.drop(['label', 'timestamp'], axis=1)
        y_test = test_df['label']
        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        return roc_auc, confusion_matrix, accuracy

    def is_worth_send_now(self, df):
        df = df.tail(1).drop(['label', 'timestamp'], axis=1)
        pred = self.model.predict(df)

        return pred[0] == 1
