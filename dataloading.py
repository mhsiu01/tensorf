import json
from tqdm import tqdm
# image libraries
from PIL import Image as Img
import cv2
# numerical
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

def load_image(name, split, indx, D):
    # Load img, remove 4th alpha channel, normalize pixels to [0.0,1.0]
    img = Img.open(f"./data/{name}/{split}/r_{indx}.png")
    if D!=800:
        img = img.resize((D,D))
    img = np.asarray(img)[:,:,:3]
    img = img / 255.0
    # Camera to world matrices + fov
    data = json.load(open(f"./data/{name}/transforms_{split}.json", 'r'))
    transform = np.asarray(data['frames'][indx]['transform_matrix'])
    fov = data['camera_angle_x']
    return img, transform, fov


# Load image and pose data
def load_data(name, split, D, debug, indices=list(range(100))):
    imgs = []
    transforms = []
    fov = None
    if debug:
        indices = [0]
    for i in indices:
        img, transform, new_fov = load_image(name, split, i, D)
        imgs.append(img)
        transforms.append(transform)
        if fov is None:
            fov = new_fov
    return imgs, transforms, fov

# def get_dataloader(indices=list(range(100))):
#     indi


# Convert camera pose to standalone rays
def pose_to_rays(pose, fov, D):
    # Homogeneous pixel coords
    v, u = np.indices((D,D))
    # y values should increase as you move up
    v = np.flip(v, axis=0)
    
    uv_homog = np.stack([u + 0.5, v + 0.5, np.ones((D,D))], axis=-1)
    # Invert intrinsics (pixel frame --> camera frame)
    focal = (D/2) / np.tan(fov/2)
    intrinsics = np.asarray([[focal, 0.0, D/2],
                           [0.0, focal, D/2],
                           [0.0, 0.0, 1.0]])

    uv_homog = np.transpose(np.reshape(uv_homog, (-1,3)))
    uv_cam = np.linalg.inv(intrinsics) @ uv_homog
    # Camera frame to world frame
    assert pose.shape==(4,4)
    origin = pose[:3,3] 
    SO3 = pose[:3,:3]
    SO3[:3,2] = -SO3[:3,2] # Unflip z-axis
    uv_world = np.transpose(SO3 @ uv_cam + origin[:,None])
    vectors_world = uv_world - origin # vector = ray - origin
    vectors_world = vectors_world / np.linalg.norm(vectors_world, axis=1)[...,None]

    origins = np.broadcast_to(origin[None,:], (D*D,3))
    rays = np.concatenate((origins,vectors_world), axis=1)
    return rays        
    
# Constructs (ray, pixel) dataset for a data split, everything as big numpy arrays
def data_to_rays(imgs, poses, fov, D):
    print("Converting camera poses to world frame rays...")
    all_rays = []
    all_pixels = []
    kept_imgs = []
    for i in tqdm(range(len(imgs)), ascii=True):
        # Convert pose matrix --> world frame ray per pixel
        rays = pose_to_rays(poses[i], fov, D) 
        all_rays.append(rays)

    # Pair each pixel value with its ray
    all_rays = np.concatenate(all_rays, axis=0)
    all_pixels = np.reshape(imgs, (-1, 3))
    print(all_rays.shape)
    print(all_pixels.shape)
    
    return all_rays, all_pixels


# Create dataloader. 
# Each ray is a concatenation of world frame origin
# and world frame unit vector direction.
# Pixels are in [0,1] range.
class NerfDataset(Dataset):
    def __init__(self, rays, pixels, device='cpu'):
        # Format to Torch tensor and as float32
        self.rays = rays.astype(np.float32)
        self.pixels = pixels
    def __len__(self):
        return self.rays.shape[0]
    def __getitem__(self, idx):
        return self.rays[idx], self.pixels[idx]