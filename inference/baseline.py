import numpy as np
from pyquaternion import Quaternion
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
        objects_in_corridor = np.logical_and(
            objects[:, 1]
            <= CORRIDOR_LENGTH + CORRIDOR_START & objects[:, 1]
            >= CORRIDOR_START,
            np.abs(objects[:, 0]) <= CORRIDOR_WIDTH / 2,
        )
        if not np.any(objects_in_corridor):
            cur_vel = input.can_bus_signals[13]
        else:
            # break
            cur_vel = 0.25  # m/s, set to low value, but not 0

        trajectory = np.zeros((6, 2))
        trajectory[:, 0] = np.arange(1, 7) * cur_vel / 2

        return UniADInferenceOutput(
            trajectory=trajectory,
            aux_outputs=output.aux_outputs,
        )


def _get_sample_input(nusc, nusc_can, scene_name, sample) -> UniADInferenceInput:
    timestamp = sample["timestamp"]
    # get the cameras for this sample
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    # ego pose via lidar sensor sample data
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sample_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    ego_translation = np.array(ego_pose["translation"])
    ego_rotation_quat = Quaternion(array=ego_pose["rotation"])
    ego2global = np.eye(4)
    ego2global[:3, 3] = ego_translation
    ego2global[:3, :3] = ego_rotation_quat.rotation_matrix

    # get cameras
    camera_tokens = [sample["data"][camera_type] for camera_type in camera_types]
    # sample data for each camera
    camera_sample_data = [nusc.get("sample_data", token) for token in camera_tokens]
    # get the camera calibrations
    camera_calibrations = [
        nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
        for cam in camera_sample_data
    ]
    # get the image filepaths
    image_filepaths = [
        nusc.get_sample_data(cam_token)[0] for cam_token in camera_tokens
    ]
    # get the camera instrinsics
    cam_instrinsics = [
        np.array(cam_calib["camera_intrinsic"]) for cam_calib in camera_calibrations
    ]
    # compute the camera2img and camera2ego transformations
    camera2img = []
    cam2global = []
    for i in range(len(camera_types)):
        # camera 2 image
        c2i = np.eye(4)
        c2i[:3, :3] = cam_instrinsics[i]
        camera2img.append(c2i)
        # camera 2 ego
        c2e = np.eye(4)
        c2e[:3, 3] = np.array(camera_calibrations[i]["translation"])
        c2e[:3, :3] = Quaternion(
            array=camera_calibrations[i]["rotation"]
        ).rotation_matrix
        # ego 2 global (for camera time)
        cam_e2g = nusc.get("ego_pose", camera_sample_data[i]["ego_pose_token"])
        cam_e2g_t = np.array(cam_e2g["translation"])
        cam_e2g_r = Quaternion(array=cam_e2g["rotation"])
        e2g = np.eye(4)
        e2g[:3, 3] = cam_e2g_t
        e2g[:3, :3] = cam_e2g_r.rotation_matrix
        # cam 2 global
        cam2global.append(e2g @ c2e)

    # load the images in rgb hwc format
    images = []
    for filepath in image_filepaths:
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    images = np.array(images)

    # get the lidar calibration
    lidar_sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_calibration = nusc.get(
        "calibrated_sensor", lidar_sample_data["calibrated_sensor_token"]
    )
    lidar_translation = np.array(lidar_calibration["translation"])
    lidar_rotation_quat = Quaternion(array=lidar_calibration["rotation"])
    lidar2ego = np.eye(4)
    lidar2ego[:3, 3] = lidar_translation
    lidar2ego[:3, :3] = lidar_rotation_quat.rotation_matrix

    lidar2global = ego2global @ lidar2ego
    global2lidar = np.linalg.inv(lidar2global)
    # the lidar2img should take into consideration that the the timestamps are not the same
    # because of this we go img -> cam -> ego -> global -> ego' -> lidar
    cam2lidar = [global2lidar.copy() @ c2g for c2g in cam2global]
    lidar2cam = [np.linalg.inv(c2l) for c2l in cam2lidar]
    lidar2img = [c2i @ l2c for c2i, l2c in zip(camera2img, lidar2cam)]

    # get the canbus signals
    pose_messages = nusc_can.get_messages(scene_name, "pose")
    can_times = [pose["utime"] for pose in pose_messages]
    assert np.all(np.diff(can_times) > 0), "canbus times not sorted"
    # find the pose that is less than the current timestamp and closest to it
    pose_idx = np.searchsorted(can_times, timestamp)
    # get the canbus signals in the correct order
    canbus_singal_order = ["pos", "orientation", "accel", "rotation_rate", "vel"]
    canbus_signals = np.concatenate(
        [
            np.array(pose_messages[pose_idx][signal_type])
            for signal_type in canbus_singal_order
        ]
    )

    return UniADInferenceInput(
        imgs=images,
        lidar_pose=lidar2global,
        lidar2img=lidar2img,
        timestamp=timestamp,
        can_bus_signals=canbus_signals,
        command=0,  # right
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = UniADRunner(
        config_path="/UniAD/projects/configs/stage2_e2e/inference_e2e.py",
        checkpoint_path="/UniAD/ckpts/uniad_base_e2e.pth",
        device=torch.device(device),
    )

    # only load this for testing
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    import matplotlib.pyplot as plt
    import cv2

    # load the first surround-cam in nusc mini
    nusc = NuScenes(version="v1.0-mini", dataroot="./data/nuscenes")
    nusc_can = NuScenesCanBus(dataroot="./data/nuscenes")
    scene_name = "scene-0103"
    scene = [s for s in nusc.scene if s["name"] == scene_name][0]
    # get the first sample in the scene
    sample = nusc.get("sample", scene["first_sample_token"])

    for i in range(60):
        inference_input = _get_sample_input(nusc, nusc_can, scene_name, sample)
        if i > 4:
            inference_input.command = 2  # straight
        plan = runner.forward_inference(inference_input)
        # plot in bev
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(inference_input.imgs[0])
        ax[0].axis("off")

        ax[1].plot(plan.trajectory[:, 0], plan.trajectory[:, 1], "r-*")
        ax[1].set_aspect("equal")
        ax[1].set_xlabel("x (m)")
        ax[1].set_ylabel("y (m)")

        # save fig
        fig.savefig(f"{scene_name}_{str(i).zfill(3)}_{sample['timestamp']}.png")
        plt.close(fig)
        if sample["next"] == "":
            break
        sample = nusc.get("sample", sample["next"])
