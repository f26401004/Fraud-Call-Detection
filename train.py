import pandas as pd
import numpy as np
import time
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class TrainModel(object):
  def __init__(self, data, test_size):
    # open log file to record current train information
    self.log = open('train.log', 'a+')
    self.log.write('\n')
    self.log.write('--------------------------------------------------\n')
    # select the feature column and target to train the model
    self.feature = ['col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col14']
    self.target = ['target']
    self.test_size = test_size

    Y = data[self.target]
    X = data[self.feature]

    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    # split test and train data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=0)
    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test

    # standardlize the train data and test data
    sc = StandardScaler()
    sc.fit(X_train)
    self.X_train_std = sc.transform(X_train)
    self.X_test_std = sc.transform(X_test)

    # log the information of the train model
    self.log_information()
  
  def log_information(self):
    self.log.write('Feature column number: %d\n' % len(self.feature))
    self.log.write('Current feature: ')
    for ele in self.feature:
      self.log.write('%s ' % ele)
    self.log.write('\n')
    self.log.write('Current target: target\n')
    self.log.write('Test size: %f\n' % self.test_size)

  def train(self):
    start_time = time.time()

    self.model = RandomForestClassifier(criterion='entropy', n_estimators=10000, random_state=50, oob_score=True, n_jobs=-1, max_features='auto')
    self.model.fit(self.X_train_std, self.Y_train)

    elapsed_time = time.time() - start_time
    self.log.write('Train elapsed time: %f\n' % elapsed_time)

  def predict(self):
    Y_pred = pd.DataFrame(self.model.predict_proba(self.X_test_std))
    result = accuracy_score(Y_pred.iloc[:, 1].round(), self.Y_test)
    print('Testing Accuracy: ', result)
    self.log.write('Testing accuracy: %f\n' % result)
    self.log.write('--------------------------------------------------\n')
  
  def predict(self, data):
    self.log.write('--------------------------------------------------\n')

    data = data[self.feature]
    # standardlize the train data and test data
    sc = StandardScaler()
    sc.fit(data)
    data = sc.transform(data)

    Y_pred = pd.DataFrame(self.model.predict_proba(data))
    Y_pred.loc[-1] = ['Probability']
    Y_pred.index = Y_pred.index + 1
    Y_pred = Y_pred.sort_index()

    # output the predict result
    Y_pred.iloc[:, 1].to_csv('./data/Submission_test.csv', sep=',', encoding='utf-8')
    