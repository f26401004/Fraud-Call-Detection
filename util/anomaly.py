import numpy as np

def find_anomalies(data):
  data_std = np.std(data).values
  data_mean = np.mean(data).values
  anomaly_cut_off = data_mean * 3

  lower_limit = data_mean - anomaly_cut_off
  upper_limit = data_mean + anomaly_cut_off

  result = []
  for outlier in range(data.shape[0]):
    if ((data.iloc[outlier].values > upper_limit).sum() > 3 or (data.iloc[outlier].values < lower_limit).sum() > 3):
      result.append(outlier)

  return result
