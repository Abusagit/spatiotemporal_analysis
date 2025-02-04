import numpy as np

def create_time_features(timestamps, unix_timeseconds, size_of_timestamps: int):
  features = np.zeros(shape=(size_of_timestamps, 9))
  min_time = np.argmin(unix_timeseconds)
  for i, day_idx in enumerate(timestamps.day_of_week):
    features[i, day_idx] = 1.0
  features[:, 7] = np.sin(2 * np.pi * (unix_timeseconds - min_time) / 86400)
  features[:, 8] = np.cos(2 * np.pi * (unix_timeseconds - min_time) / 86400)

  return features
