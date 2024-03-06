import numpy as np
import torch
from inference.runner import UniADInferenceInput, UniADInferenceOutput, UniADRunner

NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CORRIDOR_WIDTH = 5  # meters
CORRIDOR_LENGTH = 10  # meters
CORRIDOR_START = 1  # meters


class NaiveBaseline(UniADRunner):
    @torch.no_grad()
    def forward_inference(self, input: UniADInferenceInput) -> UniADInferenceOutput:
        output = super().forward_inference(input)
        objects = np.array(
            output.aux_outputs.objects_in_bev
        )  # N x 5 (x, y, w, l, yaw) (y-forward, x-right)
        objects_in_corridor_long = np.logical_and(
            objects[:, 1] <= CORRIDOR_LENGTH + CORRIDOR_START,
            objects[:, 1] >= CORRIDOR_START,
        )
        objects_in_corridor = np.logical_and(
            objects_in_corridor_long,
            np.abs(objects[:, 0]) <= CORRIDOR_WIDTH / 2,
        )

        if not np.any(objects_in_corridor):
            cur_vel = input.can_bus_signals[13]
        else:
            # break
            cur_vel = 0.0  # m/s, set to low value, but not 0

        trajectory = np.zeros((6, 2))
        trajectory[:, 1] = np.arange(1, 7) * cur_vel / 2  # y-forward

        return UniADInferenceOutput(
            trajectory=trajectory,
            aux_outputs=output.aux_outputs,
        )
