import numpy as np
from tqdm import trange

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



def normalize(features: np.ndarray, mode: str):
  new_features = features
  if features.ndim == 3:
    axes_to_normalize = (0, 1)
  else:
    axes_to_normalize = 0 # type: ignore

  match mode:
    case 'min-max':
      minimun = np.min(features, axis=axes_to_normalize, keepdims=True)
      maximum = np.max(features, axis=axes_to_normalize, keepdims=True)
      new_features -= minimun
      new_features /= (maximum - minimun)

    case 'standart':
      avg = np.average(features, axis=axes_to_normalize, keepdims=True)
      std = np.std(avg, axis=axes_to_normalize, keepdims=True)
      new_features -= avg
      new_features /= std

  return new_features



if __name__ == "__main__":
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