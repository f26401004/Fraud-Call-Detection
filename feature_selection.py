import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./data/Training_data.csv')
y = data['target'].values
data = data.drop(labels='target', axis=1)
data['col14'] = pd.Series(data['col6'] * data['col7'], index=data.index)
data['col15'] = pd.Series(data['col9'] > 20, index=data.index)
data['col16'] = pd.Series(data['col2'] > 20, index=data.index)
data['col17'] = pd.Series(data['col3'] * data['col6'], index=data.index)
data['col18'] = pd.Series(data['col9'] - data['col8'], index=data.index)

data.columns = ['', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18']

x = data.iloc[:,1:].values

sc = StandardScaler()
sc.fit(x)
x = sc.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.0, random_state=0)
feat_labels = data.columns[1:]

forest = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=50, oob_score=True, n_jobs=-1, max_features='auto')
forest.fit(x_train, y_train)

importances = forest.feature_importances_
print("Importances：", importances)
x_columns = data.columns[1:]
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
  print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
