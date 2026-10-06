import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

DATAROOT = '../nuscenes' #nuscenes dataset root
OUT_DIR = './preprocessed_data'
VERSION = 'v1.0-mini'
CHANNELS_ROOT = './channels.npy' #generated ground truth numpy file

ORIG_W, ORIG_H = 1600, 900
TARGET_W, TARGET_H = 800, 450
sx, sy = TARGET_W / ORIG_W, TARGET_H / ORIG_H

transform = transforms.Compose([
    transforms.Resize((TARGET_H, TARGET_W)),
    transforms.ToTensor(),
])

CAMERAS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]

os.makedirs(OUT_DIR, exist_ok=True)
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
all_channels = np.load(CHANNELS_ROOT)

print("Pre-processing nuScenes samples...")
for sample_idx in tqdm(range(len(all_channels))):
    sample = nusc.sample[sample_idx]
    channels = torch.tensor(all_channels[sample_idx], dtype=torch.float32)

    images = []
    intrinsics = []
    extrinsics_R = []
    extrinsics_t = []

    for cam_name in CAMERAS:
        cam_token = sample['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)
        calib_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        # Resize Image
        img_path = nusc.get_sample_data_path(cam_token)
        img = Image.open(img_path)
        images.append(transform(img))

        # Rescale Intrinsics to fit new resized images
        K = np.array(calib_sensor['camera_intrinsic'], dtype=np.float32)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        intrinsics.append(K)

        rot_cam_to_ego = Quaternion(calib_sensor['rotation']).rotation_matrix
        trans_cam_to_ego = np.array(calib_sensor['translation'], dtype=np.float32)

        R_ego2cam = rot_cam_to_ego.T
        t_ego2cam = -np.dot(R_ego2cam, trans_cam_to_ego)

        extrinsics_R.append(R_ego2cam)
        extrinsics_t.append(t_ego2cam.reshape(3, 1))

    # Pose transformations from t-1 to current t
    #check first if sample has a previous timeframe
    has_prev = (sample['prev'] != '')
    if has_prev:
        prev_sample = nusc.get('sample', sample['prev'])
        pose_curr_rec = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
        pose_prev_rec = nusc.get('ego_pose', nusc.get('sample_data', prev_sample['data']['LIDAR_TOP'])['ego_pose_token'])

        R_curr = Quaternion(pose_curr_rec['rotation']).rotation_matrix
        t_curr = np.array(pose_curr_rec['translation'])
        R_prev = Quaternion(pose_prev_rec['rotation']).rotation_matrix
        t_prev = np.array(pose_prev_rec['translation'])

        R_curr2prev = torch.tensor(R_prev.T @ R_curr, dtype=torch.float32)
        t_curr2prev = torch.tensor(R_prev.T @ (t_curr - t_prev), dtype=torch.float32).unsqueeze(-1)
    else:
        R_curr2prev = torch.zeros(3, 3)
        t_curr2prev = torch.zeros(3, 1)

    data_dict = {
        'images': torch.stack(images, dim=0),
        'channels': channels,
        'K': torch.tensor(np.stack(intrinsics), dtype=torch.float32),
        'R': torch.tensor(np.stack(extrinsics_R), dtype=torch.float32),
        't': torch.tensor(np.stack(extrinsics_t), dtype=torch.float32),
        'R_curr2prev': R_curr2prev,
        't_curr2prev': t_curr2prev,
        'has_prev': torch.tensor(has_prev),
        'sample_token': sample['token']
    }

    torch.save(data_dict, os.path.join(OUT_DIR, f"sample_{sample_idx:04d}.pt"))

print("Pre-processing complete!")