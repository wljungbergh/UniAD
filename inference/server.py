import argparse
import io
from typing import Dict, List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from .runner import (
    NUSCENES_CAM_ORDER,
    UniADInferenceInput,
    UniADRunner,
)


app = FastAPI()


class InferenceInputs(BaseModel):
    images: Dict[str, bytes]
    ego_pose: List[List[float]]  # convertable to np.array
    canbus: List[List[float]]  # convertable to np.array
    timestamp: int  # in microseconds
    command: int


class InferenceOutputs(BaseModel):
    trajectory: List[List[float]]


@app.post("/infer")
async def infer(data: InferenceInputs) -> InferenceOutputs:
    uniad_input = _build_uniad_input(data)
    uniad_output = uniad_runner.forward_inference(uniad_input)
    return InferenceOutputs(
        trajectory=uniad_output.trajectory.tolist(),
    )


def _build_uniad_input(data: InferenceInputs) -> UniADInferenceInput:
    imgs = _pngs_to_numpy([data.images[c] for c in NUSCENES_CAM_ORDER])
    ego_pose = np.array(data.ego_pose)
    lidar_pose = ego_pose  # TODO: fix
    lidar2img = ego_pose  # TODO: fix
    return UniADInferenceInput(
        imgs=imgs,
        lidar_pose=lidar_pose,
        lidar2img=lidar2img,
        can_bus_signals=np.array(data.canbus),
        timestamp=data.timestamp / 1e6,  # convert to seconds
        command=data.command,
    )


def _pngs_to_numpy(pngs: List[bytes]) -> np.ndarray:
    """Convert a list of png bytes to a numpy array of shape (n, h, w, c)."""
    imgs = []
    for png in pngs:
        img = Image.open(io.BytesIO(png))
        imgs.append(np.array(img))
    return np.stack(imgs, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    device = torch.device(args.device)

    uniad_runner = UniADRunner(args.config_path, args.checkpoint_path, device)

    uvicorn.run(app, host=args.host, port=args.port)
