# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData


class ArgoverseV1Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        am = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            kwargs, lane_data = process_argoverse(self._split, raw_path, am, self._local_radius)
            data = TemporalData(**kwargs)
            # Attach lane-level tensors to the graph sample so that they can be
            # batched together with the temporal graph representation.
            lane_counts = torch.tensor([lane_data['q_lane_type'].size(0)], dtype=torch.long)
            data['q_lane_counts'] = lane_counts
            for key, value in lane_data.items():
                data[key] = value
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx):
        saved_obj = torch.load(self.processed_paths[idx])
        if isinstance(saved_obj, TemporalData):
            data = saved_obj
        elif isinstance(saved_obj, (list, tuple)) and len(saved_obj) == 2:
            data, lane_data = saved_obj
            # Backward compatibility for previously processed artifacts.
            lane_counts = torch.tensor([lane_data['q_lane_type'].size(0)], dtype=torch.long)
            data['q_lane_counts'] = lane_counts
            for key, value in lane_data.items():
                data[key] = value
        else:
            raise RuntimeError(f'Unexpected data format in "{self.processed_paths[idx]}".')

        # Ensure q_lane_counts always exists (e.g. if processing was interrupted).
        if 'q_lane_counts' not in data:
            lane_type = data['q_lane_type']
            lane_size = lane_type.size(0) if isinstance(lane_type, torch.Tensor) else len(lane_type)
            data['q_lane_counts'] = torch.tensor([lane_size], dtype=torch.long)

        return data


def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float):
    df = pd.read_csv(raw_path)
    # filter out actors that are unseen during the historical time steps
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[: 20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)
    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]

    # make the scene centered at AV
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    # initialization
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    lane_id = torch.zeros((num_nodes, 50), dtype=torch.int64)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 20:] = True
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        for ii, temp_position in enumerate(xy):
            lane, conf, cl = am.get_nearest_centerline(np.array([temp_position[0], temp_position[1]]), city,
                                                       visualize=False)
            assert lane is not None, "lane is None"
            temp_lane_id = torch.tensor(lane.id, dtype=torch.int64)
            lane_id[node_idx, ii] = temp_lane_id

        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 20:] = True
    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]

    bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]
    positions = x.clone()
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))

    # test_x24 = x[24].numpy().copy()
    x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1: 20] - x[:, : 19])
    # test_x24_ = x[24].numpy()

    x[:, 0] = torch.zeros(num_nodes, 2)

    # get lane features at the current time step
    df_19 = df[df['TIMESTAMP'] == timestamps[19]]
    node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']]
    node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features(am, node_inds_19, node_positions_19, origin, rotate_mat, city, radius)

    y = None if split == 'test' else x[:, 20:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]
    map_features = get_map_features(am, df)

    all_lane_id = map_features["lane_ids"]
    agent_lane_id_target = target_to_idx(all_lane_id, lane_id)
    q_point_position,q_point_vector,q_lane_type, q_lane_direction,q_lane_control, q_valid_mask,q_lane_center =(
        normalize_map_feature(map_features, origin, rotate_mat))

    graph_data = {
        'x': x[:, : 20],  # [N, 20, 2]
        'positions': positions,  # [N, 50, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,

    }

    lane_data = {
        'q_point_position': q_point_position,
        'q_point_vector': q_point_vector,
        'q_lane_type': q_lane_type,
        'q_lane_direction': q_lane_direction,
        'q_lane_control': q_lane_control,
        'q_valid_mask': q_valid_mask,
        'q_lane_center': q_lane_center,
        'agent_lane_id_target': agent_lane_id_target,  # [N, 50]
    }

    return graph_data, lane_data

def target_to_idx(all_lane_id, lane_id):
    lane_id = lane_id.numpy()
    dict_all_lane_id = {int(lid): idx for idx, lid in enumerate(all_lane_id) if int(lid) > 0}
    agent_lane_id_target = np.zeros_like(lane_id)
    N, T = lane_id.shape
    for i in range(N):
        for t in range(T):
            lid = lane_id[i, t]
            if lid in dict_all_lane_id:
                agent_lane_id_target[i, t] = dict_all_lane_id[lid]
            else:
                agent_lane_id_target[i, t] = -1
    agent_lane_id_target = torch.from_numpy(agent_lane_id_target)
    return agent_lane_id_target


def normalize_map_feature(map_features, origin, rotate_mat):
    origin = origin.numpy()
    rotate_mat = rotate_mat.numpy()
    point_position = map_features['point_position']
    point_vector = map_features['point_vector']
    lane_type = map_features['lane_type']
    lane_direction = map_features['lane_direction']
    lane_control = map_features['lane_control']
    valid_mask = map_features['valid_mask']
    lane_center = map_features['lane_center']
    norm_point_position = np.matmul(point_position-origin, rotate_mat )
    norm_point_vector = np.matmul(point_vector, rotate_mat)
    norm_lane_center = np.matmul(lane_center - origin, rotate_mat)

    return (torch.from_numpy(norm_point_position).float(),
            torch.from_numpy(norm_point_vector).float(),
            torch.from_numpy(lane_type).long(),
            torch.from_numpy(lane_direction).float(),
            torch.from_numpy(lane_control).long(),
            torch.from_numpy(valid_mask).bool(),
            torch.from_numpy(norm_lane_center).float())



def interpolate_polyline(points: np.ndarray, t: int) -> np.ndarray:
    """copy from av2-api"""

    if points.ndim != 2:
        raise ValueError("Input array must be (N,2) or (N,3) in shape.")

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen: np.ndarray = np.linalg.norm(np.diff(points, axis=0), axis=1)  # type: ignore
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc: np.ndarray = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins: np.ndarray = np.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    chordlen[tbins - 1] = np.where(
        chordlen[tbins - 1] == 0, chordlen[tbins - 1] + 1e-6, chordlen[tbins - 1]
    )

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp: np.ndarray = anchors + offsets

    return points_interp

def sample_discrete_path(discrete_path, num_points: int):
    return interpolate_polyline(discrete_path, num_points)

def get_map_features(am: ArgoverseMap, df):
    sample_points = 20
    radius = 50
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[: 20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    timestamp_dfs = []
    for timestamp in timestamps:
        timestamp_df = df[df['TIMESTAMP'] == timestamp]
        # 去掉OBJECT_TYPE为AV的行
        # timestamp_df = timestamp_df[timestamp_df['OBJECT_TYPE'] != 'AV']
        timestamp_dfs.append(timestamp_df)

    present_agent_df = timestamp_dfs[19]
    node_positions = np.stack([present_agent_df['X'].values, present_agent_df['Y'].values], axis=-1)
    lane_ids = set()
    city = present_agent_df['CITY_NAME'].values[0]

    for node_position in node_positions:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    M, P = len(lane_ids), sample_points
    point_position_raw = np.zeros((M, P + 1, 2), dtype=np.float64)
    point_position = np.zeros((M, P, 2), dtype=np.float64)
    point_vector = np.zeros((M, P, 2), dtype=np.float64)
    lane_type = np.zeros((M,), dtype=np.int64)
    lane_direction = np.zeros((M,), dtype=np.float64)
    lane_control = np.zeros((M,), dtype=np.int64)
    valid_mask = np.ones((M, P), dtype=np.bool_)
    lane_center = np.zeros((M, 2), dtype=np.float64)
    for lane_id in lane_ids:
        idx = list(lane_ids).index(lane_id)
        centerline_point = am.get_lane_segment_centerline(lane_id, city)[:, : 2]
        centerline_point = sample_discrete_path(centerline_point, sample_points + 1)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)

        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')

        point_position_raw[idx] = centerline_point
        point_position[idx] = centerline_point[:-1]
        point_vector[idx] = centerline_point[1:] - centerline_point[:-1]
        lane_type[idx] = int(is_intersection)
        lane_direction[idx] = turn_direction
        lane_control[idx] = int(traffic_control)
        lane_center[idx] =  centerline_point[int(sample_points / 2)]

    map_features = {
        "point_position_raw": point_position_raw,
        "point_position": point_position,
        "point_vector": point_vector,
        "lane_type": lane_type,
        "lane_direction": lane_direction,
        "lane_control": lane_control,
        "lane_ids": list(lane_ids),
        "valid_mask": valid_mask,
        "lane_center": lane_center,
    }

    return map_features


def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    for lane_id in lane_ids:
        test = am.get_lane_segment_centerline(lane_id, city)
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)
    ddddddd = list(product(torch.arange(lane_vectors.size(0)), node_inds))
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    ddd = lane_positions.repeat_interleave(len(node_inds), dim=0)
    ccc = node_positions.repeat(lane_vectors.size(0), 1)
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors
