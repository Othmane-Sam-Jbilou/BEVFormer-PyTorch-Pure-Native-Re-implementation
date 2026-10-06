import os
import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

DATAROOT = '../nuscenes'  # nuScenes dataset root
VERSION = 'v1.0-mini'

# BEV Grid setup (meters centered at ego vehicle)
X_MIN, X_MAX = -51.2, 51.2
Y_MIN, Y_MAX = -51.2, 51.2
GRID_H, GRID_W = 200, 200  # 200x200 BEV Target Grid


def meters_to_pixels(coords_m: np.ndarray) -> np.ndarray:
    #Converts 2D (x, y) coordinates in meters to (col, row) pixel indices.
    col = (coords_m[:, 0] - X_MIN) / (X_MAX - X_MIN) * (GRID_W - 1)
    row = (coords_m[:, 1] - Y_MIN) / (Y_MAX - Y_MIN) * (GRID_H - 1)
    return np.column_stack([col, row]).astype(np.int32)


def get_category_channel(category_name: str) -> int:
    """
    Maps nuScenes categories to 4 object channels (1-4).
    Channel 0 is reserved for drivable area.
    """
    if 'cycle' in category_name or 'bicycle' in category_name or 'motorcycle' in category_name:
        return 3  # Two-wheelers
    elif 'vehicle' in category_name:
        return 1  # Cars, trucks, buses, trailers
    elif 'human' in category_name or 'pedestrian' in category_name:
        return 2  # Pedestrians
    elif 'barrier' in category_name or 'trafficcone' in category_name or 'object' in category_name:
        return 4  # Static objects/barriers
    return -1


def paint_drivable_area(bev_channel: np.ndarray, nusc: NuScenes, sample: dict, ego_translation: np.ndarray, ego_rotation: Quaternion):
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_name = log['location']
    nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=map_name)

    # Query map geometries around ego vehicle (120m x 120m patch)
    patch_box = (ego_translation[0], ego_translation[1], 120, 120)
    map_geoms = nusc_map.get_map_geom(patch_box, patch_angle=0, layer_names=['drivable_area'])

    R_global_to_ego = ego_rotation.rotation_matrix[:2, :2]

    polys = []
    for layer_name, geoms in map_geoms:
        for g in geoms:
            if hasattr(g, 'geom_type'):
                if g.geom_type == 'MultiPolygon':
                    polys.extend(list(g.geoms))
                elif g.geom_type == 'Polygon':
                    polys.append(g)

    # Paint all polygons into Channel 0
    for poly in polys:
        ext_coords = np.array(poly.exterior.coords)[:, :2]
        pts_ego = (ext_coords - ego_translation[:2]) @ R_global_to_ego.T
        pts_px = meters_to_pixels(pts_ego)
        cv2.fillPoly(bev_channel, [pts_px], 1.0)


def generate_bev_ground_truth(nusc: NuScenes, sample_idx: int) -> np.ndarray:
    sample = nusc.sample[sample_idx]

    # Fetch LiDAR pose
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation']).inverse

    # Output grid shape: (5, 200, 200)
    all_channels = np.zeros((5, GRID_H, GRID_W), dtype=np.float32)

    # Channel 0: Drivable Area
    paint_drivable_area(all_channels[0], nusc, sample, ego_translation, ego_rotation)

    # Channels 1-4: Object Bounding Boxes
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        channel_idx = get_category_channel(ann['category_name'])
        if channel_idx < 0:
            continue

        # Transform bounding box to Ego Frame
        box_global = nusc.get_box(ann_token)
        box_ego = box_global.copy()
        box_ego.translate(-ego_translation)
        box_ego.rotate(ego_rotation)

        # Extract 2D bottom corners in Ego frame
        corners_ego = box_ego.bottom_corners()[:2, :].T
        corners_px = meters_to_pixels(corners_ego)

        cv2.fillPoly(all_channels[channel_idx], [corners_px], 1.0)

    return all_channels


if __name__ == '__main__':
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    all_samples = []
    num_samples = len(nusc.sample)
    print(f"Generating BEV GT maps for {num_samples} samples...")

    for i in range(num_samples):
        if (i + 1) % 50 == 0 or i == num_samples - 1:
            print(f"Processing sample [{i + 1}/{num_samples}]")
        channels = generate_bev_ground_truth(nusc, sample_idx=i)
        all_samples.append(channels)

    all_samples = np.stack(all_samples, axis=0)  # Shape: (N, 5, 200, 200)
    np.save('channels.npy', all_samples)
    print(f"Saved dataset successfully to 'channels.npy' with shape {all_samples.shape}.")