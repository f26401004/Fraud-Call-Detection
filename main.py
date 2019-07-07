import pandas as pd
import numpy as np
from train import TrainModel
from util.anomaly import find_anomalies
from sklearn.preprocessing import StandardScaler

def main():
  X = pd.read_csv('./data/Training_data.csv')

  # observe the data distribution
  print(X[X['target'] == 1].sample(5))
  print(X[X['target'] == 0].sample(5))
  print(X['target'].value_counts())

  # remove the derviation value in non-fraud domain
  X = X.drop(X.index[find_anomalies(X)])

  X['col14'] = pd.Series(X['col7'] * X['col6'], index=X.index)
  
  # X_fraud = X[X['target'] == 1].sample(1000)
  # X_non_fraud = X[X['target'] == 0].sample(1000)
  # X_shuffle = X_fraud.append(X_non_fraud)
  # X_shuffle = X_shuffle.reindex(np.random.permutation(X_shuffle.index))

  model = TrainModel(X, 0.0001)
  model.train()

  # prepare testing data
  X_pred = pd.read_csv('./data/Testing_data.csv')
  X_pred['col14'] = pd.Series(X_pred['col7'] * X_pred['col6'], index=X_pred.index)

  model.predict(X_pred)


if __name__ == '__main__':
  main()