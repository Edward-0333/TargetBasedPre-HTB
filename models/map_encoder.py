import torch
import torch.nn as nn

from models import FourierEmbedding
from models import PointsEncoder
from torch.nn.utils.rnn import pad_sequence


class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        use_lane_boundary=False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.use_lane_boundary = use_lane_boundary
        self.polygon_channel = polygon_channel
        self.polygon_encoder = PointsEncoder(self.polygon_channel, dim)
        self.speed_limit_emb = FourierEmbedding(1, dim, 64)
        self.control_emb = nn.Embedding(2, dim)
        self.type_emb = nn.Embedding(2, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.direction_emb = nn.Embedding(3, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

    def forward(self, data):
        lane_type = data['q_lane_type']
        point_position = data['q_point_position']
        point_vector = data['q_point_vector']
        lane_control = data['q_lane_control']
        lane_direction = data['q_lane_direction']
        lane_center = data['q_lane_center']
        valid_mask = data['q_valid_mask']
        lane_counts = data.get('q_lane_counts', None)

        if lane_counts is None:
            lane_counts = torch.tensor([lane_type.size(0)], device=lane_type.device)
        if lane_counts.dim() == 0:
            lane_counts = lane_counts.unsqueeze(0)
        lane_counts = lane_counts.to(lane_type.device)

        polygon_feature = torch.cat(
            [
                point_position - lane_center.unsqueeze(1),
                point_vector,
            ],
            dim=-1,
        )

        x_polygon = self.polygon_encoder(polygon_feature, valid_mask)

        x_type = self.type_emb(lane_type)
        x_control = self.control_emb(lane_control)
        x_direction = self.direction_emb(lane_direction.long())

        x_polygon = x_polygon + x_type + x_control + x_direction

        lane_features = list(x_polygon.split(lane_counts.tolist()))
        valid_mask = list(valid_mask.split(lane_counts.tolist()))
        lane_features = pad_sequence(lane_features, batch_first=True, padding_value=0.0)
        valid_mask = pad_sequence(valid_mask, batch_first=True, padding_value=False)
        return lane_features, valid_mask
