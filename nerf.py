# Imports
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image as Img
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

from IPython import embed
from tqdm import tqdm



# Preprocessing

# Get all pose and rgb file names for a data split
def get_files(split, D):
    if split=='train':
        N = 100
        prefix = '0_train'
    if split=='val':
        N = 100
        prefix = '1_val'
    if split=='test':
        N = 200
        prefix = '2_test'
    # Get file names except for file type suffix
    IDs = [f"{prefix}_{str(i).rjust(4,'0')}" for i in range(0,N)]
    # Load poses and imgs
    poses = [np.loadtxt(f"./bottles/pose/{ID}.txt") for ID in tqdm(IDs, ascii=True)]
    imgs = [Img.open(f"./bottles/rgb/{ID}.png").convert("RGB") for ID in tqdm(IDs, ascii=True)] # Remove 4th alpha channel
    return poses, imgs
    
# Convert camera pose to standalone rays
def pose_to_rays(pose, D, split):
    # Homogeneous pixel coords
    v, u = np.indices((D,D))
    uv_homog = np.stack([u + 0.5, v + 0.5, np.ones((D,D))], axis=-1)
    # Invert intrinsics (pixel frame --> camera frame)
    intrinsics = np.loadtxt("./bottles/intrinsics.txt")
    if D!=800: # Change intrinsics matrix if using lower image resolution
        scale = D / 800
        intrinsics[0:2,0:3] = intrinsics[0:2,0:3] * scale
    uv_homog = np.transpose(np.reshape(uv_homog, (-1,3)))
    uv_cam = np.linalg.inv(intrinsics) @ uv_homog
    # Camera frame to world frame
    assert pose.shape==(4,4)
    origin = pose[:3,3] 
    SO3 = pose[:3,:3]
    uv_world = np.transpose(SO3 @ uv_cam + origin[:,None])
    vectors_world = uv_world - origin # vector = ray - origin
    vectors_world = vectors_world / np.linalg.norm(vectors_world, axis=1)[...,None]

    origins = np.broadcast_to(origin[None,:], (D*D,3))
    rays = np.concatenate((origins,vectors_world), axis=1)
    return rays    
    
# Constructs (ray, pixel) dataset for a data split
def get_dataset(split, D):
    # Get list of pose matrices and list of RGB PIL images. Reshape as needed.
    print(f"Loading {split} split...")
    poses, imgs_complete = get_files(split, D)
    if split=='train' and D!=800:
        imgs_complete = [img.resize((D,D)) for img in imgs_complete]

    print("Converting camera poses to world frame rays...")
    all_rays = []
    all_pixels = []
    kept_imgs = []
    for i in tqdm(range(len(imgs_complete)), ascii=True):
        # Only use a few images for validation (to save time)
        if split=='val' and (i not in [35,70]): # i = 35, 70
            continue
        # Convert pose matrix --> world frame ray per pixel
        rays = pose_to_rays(poses[i], D, split) 
        all_rays.append(rays)
        # PIL to numpy, normalize to [0,1] range, flatten image pixels.
        pixels = np.reshape(np.asarray(imgs_complete[i])/255.0, (-1,3))
        all_pixels.append(pixels)
        # Only keep imgs that we are actually using
        kept_imgs.append(imgs_complete[i])

     # Pair each pixel value with its ray
    all_rays = np.concatenate(all_rays, axis=0) # [(800*800,3+3),...x100] --> (100*800*800,3+3)
    all_pixels = np.concatenate(all_pixels, axis=0)
    print(all_rays.shape)
    print(all_pixels.shape)
    return all_rays, all_pixels, kept_imgs


# Training

# Create dataloader. 
# Each ray is a concatenation of world frame origin
# and world frame unit vector direction.
# Pixels are in [0,1] range.
class NerfDataset(Dataset):
    def __init__(self, rays, pixels, device='cpu'):
        # Format to Torch tensor and as float32
        self.rays = torch.as_tensor(rays, dtype=torch.float32, device=device)
        self.pixels = torch.as_tensor(pixels, dtype=torch.float32, device=device)
    def __len__(self):
        return self.rays.shape[0]
    def __getitem__(self, idx):
        return self.rays[idx], self.pixels[idx]

# Define MLP model

# Basic set of layers, repeat to increase model depth arbitrarily
class fc_layer(nn.Module):
    def __init__(self, channels, p):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
    def forward(self, input):
        return self.layer(input)

class Nerf(nn.Module):
    def __init__(self, L_xyz, L_directions, N_layers, channels, p):
        super().__init__()
        self.xyz_width = 2*L_xyz*3
        self.directions_width = 2*L_directions*3
        self.N_layers = N_layers
        self.channels = channels
        self.dropout = p

        # Input layer
        input0 = nn.Sequential(
            nn.Linear(self.xyz_width, channels),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        # Hidden layers before skip connection
        self.hidden1 = nn.ModuleList([input0] + [fc_layer(channels,p) for i in range(N_layers-1)])
        # Skip connection
        skip = nn.Sequential(
            nn.Linear(self.xyz_width + channels, channels),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        # Hidden layers after skip connection
        self.hidden2 = nn.ModuleList([skip] + [fc_layer(channels,p) for i in range(N_layers-1)])

        # Predicts a non-negative density using output of hidden2.
        # Density is a non-negative real number.
        self.density_layer = nn.Sequential(
            nn.Linear(channels, 1),
            nn.ReLU()
        )
        self.no_activation_linear = nn.Linear(channels, channels)
        # Color/radiance is in [0,1], for three color channels.
        self.color_layer = nn.Sequential(
            nn.Linear(channels+self.directions_width, int(channels/2)),
            nn.ReLU(),
            nn.Linear(int(channels/2),3),
            nn.Sigmoid()
        )

    def forward(self, xyz, directions):
        # First half of layers
        x = xyz
        for i, l in enumerate(self.hidden1):
            x = l(x)

        # Skip connection to location input
        x = torch.cat((x,xyz), dim=1)
        
        # Second half of layers
        for i, l in enumerate(self.hidden2):
            x = l(x)
    
        # Predict density scalar. First element will be used for sigma.
        sigma = self.density_layer(x)
        # Concat view direction to x, which is just feature vector pre-density_layer
        x = self.no_activation_linear(x)
        x = torch.cat((x,directions), dim=1)
        # Predict RGB radiance
        rgb = self.color_layer(x)
        return sigma,rgb


# ## Point sampling

# Overall pipeline: Ray --> uniform steps --> perturb --> xyz --> sine/cosine encode --> pass thru MLP

# Returns points uniformly sampled from evenly-spaced buckets in [t_near, t_far]
def integration_times(N_samples, t_near=0, t_far=5):
    # eg. suppose N_samples=64. Then there are 64 bucket_points...
    bucket_points = torch.linspace(start=t_near, end=t_far, steps=N_samples)
    bucket_width = bucket_points[1] - bucket_points[0]
    assert bucket_width == (t_far-t_near)/(N_samples-1)
    # ...with 63 uniform samples
    u = torch.rand(N_samples - 1)
    # ...which replace the middle 62 points
    t_sampled = bucket_points[0:-1] + u*bucket_width
    times = torch.cat((torch.tensor(t_near)[None], t_sampled, torch.tensor(t_far)[None]))
    # ...for a total of 65 points. But the 65th point won't go thru MLP.
    return times

# Given rays and integration times, return points in xyz space and view direction.
# Assumes batch of rays, and single vector of times.
def rays_to_points(rays, t):
    origins, vectors = rays[:,0:3], rays[:,3:6]
    xyz = origins[:,None,:] + vectors[:,None,:]*t[None,:,None]
    directions = torch.broadcast_to(vectors[:,None,:], xyz.shape)
    return xyz,directions

# Encodes batch of vectors
def encode(xyz, directions, L_xyz=2, L_directions=3):
    xyz_encoded = []
    for i in range(L_xyz):
        xyz_encoded.append(torch.sin((2**i)*xyz))
        xyz_encoded.append(torch.cos((2**i)*xyz))
        
    directions_encoded = []
    for i in range(L_directions):
        directions_encoded.append(torch.sin((2**i)*directions))
        directions_encoded.append(torch.cos((2**i)*directions))
    
    xyz_encoded = torch.cat(xyz_encoded, dim=-1)
    directions_encoded = torch.cat(directions_encoded, dim=-1)
    return xyz_encoded, directions_encoded


# Raymarching function. Assumes sigma and rgb are shaped as batch of per-ray values.
# Integration times + radiances + densities --> pixel value
def raymarch(t,sigma,rgb):
    t_widths = t[0:-1] - t[1:]
    transmits = torch.exp(-1*sigma*t_widths[None,:])
    alphas = 1.0 - transmits
    
    T = torch.cumprod(transmits + 1e-10, dim=1)
    weights = T*alphas
    renders = torch.sum(weights[:,:,None]*rgb, dim=1)
    bkgd = transmits[:,-1]

    renders = torch.clip(renders + bkgd[:,None], min=0.0, max=1.0)

    return renders




# Training

def forward_pass(rays, pixels, mlp, device, N_samples, L_xyz, L_directions):
    N_rays = rays.shape[0]

    # Sample points along ray between t_near and t_far in integration time
    t = integration_times(N_samples)
    t = t.to(device)
    # Convert to xyz+direction in world frame
    xyz,directions = rays_to_points(rays, t)
    # Encode xyz points with Fourier. 
    xyz,directions = encode(xyz,directions,L_xyz,L_directions)
    # Shape is (batch_size, N_samples+1, 2*L*3)

    # Remove point at t_far for each ray, reshape to batch of sample points
    xyz = xyz[:,:-1,:]
    directions = directions[:,:-1,:]
    xyz = torch.reshape(xyz, (-1,2*L_xyz*3))
    directions = torch.reshape(directions, (-1,2*L_directions*3))

    # Pass through MLP
    sigma,rgb = mlp(xyz,directions)
    # Reshape to batch of rays and raymarch
    sigma = torch.reshape(sigma, (N_rays, N_samples))
    rgb = torch.reshape(rgb, (N_rays, N_samples, -1))
    renders = raymarch(t,sigma,rgb)
    return renders

def validation(dataloader_val, device, mlp, N_samples, L_xyz, L_directions, D, epoch, label_imgs, writer):
    mlp.eval()
    with torch.no_grad():
        # Forward pass thru validation set
        renders_all = []
        losses = []
        for iteration, (rays, pixels) in enumerate(tqdm(dataloader_val, ascii=True)):
            rays = rays.to(device)
            pixels = pixels.to(device)
            with torch.cuda.amp.autocast():
                renders = forward_pass(rays, pixels, mlp, device, N_samples, L_xyz, L_directions) 
                renders_all.append(renders)
                losses.append(F.mse_loss(renders, pixels))
        val_loss = sum(losses)/len(losses)
        print(f"val #{epoch} loss = {val_loss}")
        
        # Process renders into image
        renders_all = torch.cat(renders_all, dim=0)
        renders_all = torch.reshape(renders_all, (-1,D,D,3))*255
        renders_all = renders_all.cpu().numpy()
        renders_all = renders_all.astype(np.uint8)
        writer.add_image(f"val_epoch{epoch}", np.transpose(renders_all,(0,3,1,2)), dataformats='NCHW')

        # Calculate PSNR using cv2 format
        PSNRs = []
        for i,label_img in enumerate(label_imgs):
            # Convert PIL images to cv2 color mode
            render_img = cv2.cvtColor(renders_all[i], cv2.COLOR_RGB2BGR) # numpy RGB --> cv2 BGR
            label_img = cv2.cvtColor(np.asarray(label_img), cv2.COLOR_RGB2BGR) # PIL RGB --> numpy --> cv2 BGR
            # psnr score
            psnr = cv2.PSNR(render_img, label_img)
            PSNRs.append(psnr)
            print(f"        psnr = {psnr}")
            # cv2.imwrite(f"./rendered_imgs/epoch{epoch}img{i}psnr{round(psnr)}.png", render_img)
        val_psnr = sum(PSNRs)/len(PSNRs)
    
    writer.add_scalar("loss/val", val_loss, epoch)
    writer.add_scalar("loss/psnr", val_psnr, epoch)
    return val_loss, val_psnr




def main(D, t_near, t_far, N_samples, layers, channels, N_epochs, lr,
         batch_size, device, L_xyz, L_directions,
         start_epoch, p, gamma, logdir, checkpointdir):
    # Preprocessing
    rays_train, pixels_train, imgs_train = get_dataset('train', D)
    rays_val, pixels_val, imgs_val = get_dataset('val', 800)
    # Dataloader
    dataset_train = NerfDataset(rays_train, pixels_train)
    dataset_val = NerfDataset(rays_val, pixels_val)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    # Training objects
    if start_epoch==1:
        mlp = Nerf(L_xyz, L_directions, N_layers=N_layers, channels=channels, p=p).to(device)
        optimizer = torch.optim.Adam(
            mlp.parameters(),
            lr=lr,
            betas=(0.9, 0.999), 
            eps=1e-07, 
            weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma, last_epoch=-1
        )
        scaler = torch.cuda.amp.GradScaler()
    else:
        checkpoint = torch.load(f"./checkpoints/{start_epoch-1}.pth")
        mlp = checkpoint['model'].to(device)
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        scaler = checkpoint['scaler']
        logdir = checkpoint['logdir']
    writer = SummaryWriter(logdir)
    print(mlp)
    mlp.train()

    # Training loop
    for epoch in range(start_epoch,N_epochs+1):
        # Training loop, one epoch
        mlp.train()
        losses = []
        for iteration, (rays, pixels) in enumerate(tqdm(dataloader_train, ascii=True)):
            rays = rays.to(device)
            pixels = pixels.to(device)
            with torch.cuda.amp.autocast():
                renders = forward_pass(rays, pixels, mlp, device, N_samples, L_xyz, L_directions)
                loss = F.mse_loss(renders, pixels)

            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # If not using mixed precision:
            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)
        print(f"train #{epoch} loss = {train_loss}")
        scheduler.step()
        writer.add_scalar("loss/train", train_loss, epoch)

        # Validation
        if epoch%5==0:
            # Rendering on validation images
            val_loss, val_psnr = validation(dataloader_val, device, mlp, N_samples, L_xyz, L_directions, 800, epoch, imgs_val, writer)
            # Save checkpoint
            checkpoint = {
                'model': mlp,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'scaler': scaler,
                'logdir': logdir,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_psnr': val_psnr
            }
            torch.save(checkpoint, f"{checkpointdir}{epoch}.pth")
            print(f"Saved checkpoint for epoch #{epoch}.")
            print()
    writer.close()


if __name__=='__main__':
    # Set training hyperparameters
    t_near = 0
    t_far = 5
    device = 'cuda'
    # Resolution and rays
    D = 200
    N_samples = 256
    batch_size = 512
    p = 0.0
    channels = 256
    N_layers = 256
    # Embedding
    L_xyz = 10
    L_directions = 4
    # Training schedule
    N_epochs = 25
    lr = 0.0005
    decay = 0.5
    start_epoch = 1
    run = "run_09a"    

    # Create directories for Tensorboard and checkpoints
    logdir = f"./runs/{run}/lr{lr}_decay{decay}_Nepochs{N_epochs}_D{D}_batch{batch_size}_p{p}_samples{N_samples}_Nlayers{N_layers}_channels{channels}_L{L_xyz},{L_directions}"
    checkpointdir = f"./checkpoints/{run}/"
    Path(checkpointdir).mkdir(parents=True, exist_ok=True)
    print(f"{checkpointdir=}")
    print(f"{logdir=}")
    # Set lr decay
    gamma = decay**(1.0/N_epochs)
    print(f"{gamma=}")
    print(f"{N_epochs=}")
    print(f"{lr=}")
    print(f"{N_layers=}")
    print(f"{channels=}")
    main(D, t_near, t_far, N_samples, N_layers, channels, N_epochs, lr, batch_size, device, L_xyz, L_directions, start_epoch, p, gamma, logdir, checkpointdir)
