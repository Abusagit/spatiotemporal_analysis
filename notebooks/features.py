import numpy as np
from tqdm import trange
import pandas as pd


def create_time_features(timestamps, unix_timeseconds, size_of_timestamps: int):
  features = np.zeros(shape=(size_of_timestamps, 9))
  min_time = np.argmin(unix_timeseconds)
  for i, day_idx in enumerate(timestamps.day_of_week):
    features[i, day_idx] = 1.0
  features[:, 7] = np.sin(2 * np.pi * (unix_timeseconds - min_time) / 86400)
  features[:, 8] = np.cos(2 * np.pi * (unix_timeseconds - min_time) / 86400)

  return features


def create_moment_agregated_features(graph, nodes_number: int, features: np.ndarray, modes: list[str]):
  # final_answer = np.zeros((len(modes), nodes_number, features.shape[1]))
  final_answer = []

  for i, mode in enumerate(modes):
    mode_answer = np.zeros((nodes_number, features.shape[1]))
    
    for start_node in range(len(graph)):
      indices_of_neibours = np.nonzero(graph[start_node])[0]
      if len(indices_of_neibours) == 0:
        mode_answer[start_node] = np.zeros(features.shape[1])
      else:
        match mode:
          case "mean":
            mode_answer[start_node] = features[indices_of_neibours].mean(0)
          case "min":
            mode_answer[start_node] = features[indices_of_neibours].min(0)
          case "median":
            mode_answer[start_node] = np.median(features[indices_of_neibours], 0)
          case "max":
            mode_answer[start_node] = features[indices_of_neibours].max(0)
          case _:
            raise ValueError(f"Mode `{mode}` is not supported!")
      
    final_answer.append(mode_answer)

  return np.concatenate(final_answer, axis = 1)


def create_features(number_of_timestamps, graph, nodes_number: int, features: np.ndarray, modes: list[str]):
  result = []
  for i in trange(number_of_timestamps):
    result.append(create_moment_agregated_features(graph, nodes_number, features[i], modes))
  return np.stack(result, axis = 0)



def normalize(features: np.ndarray, mode = 'standart'):
  new_features = np.copy(features)
  if features.ndim == 3:
    axes_to_normalize = (0, 1)
  else:
    axes_to_normalize = 0 # type: ignore

  match mode:
    case 'min-max':
      minimun = np.min(features, axis=axes_to_normalize, keepdims=True)
      maximum = np.max(features, axis=axes_to_normalize, keepdims=True)
      print(f"minimun: {minimun}")    
      print(f"minimun: {maximum}") 
      new_features -= minimun
      range_ = maximum - minimun
      range_ = np.where(np.abs(range_ - 0) < 1e-6, 1, range_)
      new_features /= range_

    case 'standart':
      avg = np.average(features, axis=axes_to_normalize, keepdims=True)
      print(avg)
      std = np.std(features, axis=axes_to_normalize, keepdims=True)
      print(std)
      std = np.where(np.abs(std - 0) < 1e-6, 1, std)
      print(std)
      new_features -= avg
      new_features /= std

  return new_features


from pathlib import Path




def read_dataset_file(path: Path) -> np.lib.npyio.NpzFile:
    return np.load(file=path, allow_pickle=True)


def distr_of_jams(dataset, timestamps,  node, start = 0, end = 288):
  distribution = np.array([0] * 288)
  for i in range(start, end):
      if dataset[i][node] < 15:
          timestamp_cur = timestamps[i]
          min_cur = timestamp_cur.hour * 60 + timestamp_cur.minute
          distribution[min_cur//5]+=1
  return distribution


def jams_propability(dataset, timestamps, node, start = 0, end = 288):
  distribution = distr_of_jams(dataset, timestamps, node, start, end)
  distribution_sum = np.sum(distribution)
  distribution_sum = np.where(np.abs(distribution_sum - 0) < 1e-8, 1, distribution_sum)
  probability = distribution / (distribution_sum + 1e-6)
  return probability
   

def create_all_features(graph: list[list], timestamps, target, mode: str, nodes_number = 207):
  number_of_timestamps = len(timestamps)
  all_probabilities = np.zeros((nodes_number, 288))
  for node in range(nodes_number):
    all_probabilities[node] = jams_propability(target, timestamps, node, 0, number_of_timestamps)
  
  print((all_probabilities>0).sum(0))

  probabilities = []
  for _ in trange(number_of_timestamps // 288):
    probabilities.append(all_probabilities)
  
  all_prob = np.concatenate(probabilities, axis=1).T
  all_prob = np.nan_to_num(all_prob, 0)
  print(all_prob.shape)
  features = target[:, :, None]
  features = np.nan_to_num(features, 0)
  print(features.shape)

  features_concatenated_with_jams = np.concatenate([features, all_prob[:, :, None]], axis=-1)
  print(features_concatenated_with_jams.shape)

  graph_view = np.zeros((nodes_number, nodes_number))

  for node_1, node_2 in graph:
      graph_view[node_2][node_1] = 1

  features_matrix = create_features(len(timestamps), graph_view, 207, features_concatenated_with_jams, ['mean'])
  
  return features_matrix

def region_features():
  districts = [[0, 13, 36, 37, 51, 54, 58, 61, 62, 67, 111, 112, 114, 115, 116, 117, 118, 140, 142, 143, 145, 190, 194, 199],
               [1, 2, 7, 11, 18, 21, 27, 28, 35, 46, 50, 55, 66, 78, 79, 85, 92, 105, 106, 107, 108, 121, 123, 126, 132, 135, 177, 189, 200],
               [3, 4, 5, 6, 12, 15, 16, 17, 19, 20, 22, 29, 30, 32, 33, 38, 39, 40, 48, 57, 65, 68, 70, 71, 74, 80, 91, 93, 94, 96, 97, 98, 102, 103, 119, 127, 128, 136, 138, 144, 154, 155, 157, 159, 160, 161, 162, 163, 166, 175, 187, 188, 191, 192, 193, 195, 196, 198, 205, 206],
               [8, 14, 34, 59, 77, 84, 88, 89, 104, 151, 182, 185, 186],
               [9, 41, 86, 87, 100, 130, 131, 146, 148, 150, 172, 180, 181, 197, 204],
               [10, 31, 83, 90, 99, 122, 149, 156, 176],
               [23, 25, 49, 56, 64, 95, 101, 109, 120, 124, 125, 129, 133, 134, 139, 147, 165, 170, 174, 183, 184],
               [24, 42, 44, 45, 53, 110, 152, 153, 167, 168, 169, 171, 173, 178, 179, 201, 202, 203],
               [43, 47, 52, 60, 63, 69, 72, 73, 75, 76, 81, 82, 113, 137, 141, 158, 164]]
  
  district_feature = [[j , i] for i in range(len(districts)) for j in districts[i]] + [[26, -1]]  # for undefined node
  sorted_district_feature = sorted(district_feature, key=lambda x: x[0])
  list_district_feature = [i for _, i in sorted_district_feature]
  enum_district_feature = []
  for i in range(len(list_district_feature)):
    enum_district_i = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    district_id = list_district_feature[i]
    if district_id != -1:
      enum_district_i[district_id] = 1
    enum_district_feature.append(enum_district_i)
  print(enum_district_feature)
  return np.array(enum_district_feature)

def create_all_normal_features(graph: list[list], mode: str):
  DATA_DIR = Path("../data/")
  dataset = np.load(file= DATA_DIR/"metr_la_new.npz", allow_pickle=True)
  target = dataset['targets']
  timestamps = pd.date_range(start=dataset["first_timestamp_datetime"].item(),
                           end=dataset["last_timestamp_datetime"].item(),
                           freq="5min",
                           )
  features = create_all_features(graph, timestamps, target, mode)
  norm_features = normalize(features, 'standart')
  region_feature = region_features()[None, ...]
  repeat_region_feature = region_feature.repeat(norm_features.shape[0], axis = 0)
  
  time_features_expanded = create_time_features(
      timestamps=pd.date_range(start=dataset["first_timestamp_datetime"].item(),
                              end=dataset["last_timestamp_datetime"].item(),
                              freq="5min",
                            ),
      unix_timeseconds=dataset["unix_timestamps"],
      size_of_timestamps=dataset["targets"].shape[0]
  )[:, None, :].repeat(dataset["targets"].shape[1], 1)

  return np.concatenate([norm_features, repeat_region_feature, time_features_expanded], axis = -1).astype(np.float32)



if __name__ == "__main__":
  print(region_features())

  create_all_normal_features([], mode='min')
  """metr_la = np.load(file= DATA_DIR/"metr_la_new.npz", allow_pickle=True)
  targets = metr_la['targets']
  ver_index = 5
  start = 0
  end = len(targets)

  timestamps = pd.date_range(start=metr_la["first_timestamp_datetime"].item(),
                           end=metr_la["last_timestamp_datetime"].item(),
                           freq="5min",
                           )

  create_all_features([], timestamps, targets, mode='min')

  graph = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [1, 1 ,0]])


  fake_features = np.array([[1, 2],
                            [1, 1],
                            [3, 4]])


  features_ground_truth = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 1, 2, 1, 2, 1, 2],
    [1, 3/2, 1, 2, 1, 3/2, 1, 1]
  ])

  result = create_moment_agregated_features(graph, 3, fake_features, modes=['mean', 'max', 'median', 'min'])
  assert np.allclose(result, features_ground_truth)

  fake_features_multi = np.array([
    [[1.], [1.], [3.]],
    [[2.], [1.], [4.]],
  ])

  features_multi_ground_truth = np.array([
    [[0.], [1.], [1.]],
    [[0.], [2.], [2.]],
  ])

  result_multi_timestamps = create_features(features_multi_ground_truth.shape[0], graph, 3, fake_features_multi, ['max'])
  print(result_multi_timestamps)
  # breakpoint()
  assert np.allclose(result_multi_timestamps, features_multi_ground_truth)

  
"""