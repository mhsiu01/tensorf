import numpy as np
import torch
import pdb

# ## Point sampling

# Overall pipeline: Ray --> uniform steps --> perturb --> xyz --> sine/cosine encode --> pass thru MLP

# See: https://github.com/pytorch/pytorch/issues/102208#issuecomment-1568998834
def batch_linspace(start, end, steps, device):
    # steps: Total points in interval, from start to end is actually steps-1 increments
    out = start[:,None] + torch.arange(0, steps, device=device)[None,:] * ((end-start)/(steps-1))[:,None]
    return out
    
# Returns points uniformly sampled from evenly-spaced buckets in [t_near, t_far]
def integration_times(num_samples, t_near, t_far, device=None, bespoke_t=False):
    # eg. suppose num_samples=64. Then there are 64 bucket_points...
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if bespoke_t:
        bucket_points = batch_linspace(t_near, t_far, num_samples, device)
        bucket_width = bucket_points[:,1] - bucket_points[:,0]
        u = torch.rand(num_samples - 1, device=device)
        # Broadcast to dimensions (batch rays, num_samples):
        t_sampled = bucket_points[:,0:-1] + u[None,:]*bucket_width[:,None]
        # Concatenate along num_samples dimension
        times = torch.cat((torch.tensor(t_near, device=device)[:,None],
                           t_sampled,
                           torch.tensor(t_far, device=device)[:,None]),
                          dim=1)
    else:
        bucket_points = torch.linspace(start=t_near, end=t_far, steps=num_samples, device=device)
        bucket_width = bucket_points[1] - bucket_points[0]
        assert torch.abs(bucket_width - (t_far-t_near)/(num_samples-1)) < 1e-4#bucket_width == (t_far-t_near)/(num_samples-1)
        # ...with 63 uniform samples
        u = torch.rand(num_samples - 1, device=device)
        # ...which replace the middle 62 points
        t_sampled = bucket_points[0:-1] + u*bucket_width
        times = torch.cat((torch.tensor(t_near, device=device)[None],
                           t_sampled,
                           torch.tensor(t_far, device=device)[None]))
    # ...for a total of 65 points. But the 65th point won't go thru MLP.
    return times

# Given rays and integration times, return points in xyz space and view direction.
# Assumes batch of rays, and single vector of times.
def rays_to_points(rays, t):
    origins, vectors = rays[:,0:3], rays[:,3:6]
    xyz = origins[:,None,:] + vectors[:,None,:]*t[:,:,None]
    directions = torch.broadcast_to(vectors[:,None,:], xyz.shape)
    return xyz,directions

# Encodes batch of vectors
def encode(data, L):
    encoded = []
    for i in range(L):
        encoded.append(torch.sin((2**i)*data))
        encoded.append(torch.cos((2**i)*data))
    encoded = torch.cat(encoded, dim=-1)
    return encoded

def encode_all(xyz, directions, L_xyz=2, L_directions=3):
    xyz_encoded = encode(xyz, L_xyz)
    directions_encoded = encode(directions, L_directions)
#     xyz_encoded = []
#     for i in range(L_xyz):
#         xyz_encoded.append(torch.sin((2**i)*xyz))
#         xyz_encoded.append(torch.cos((2**i)*xyz))
        
#     directions_encoded = []
#     for i in range(L_directions):
#         directions_encoded.append(torch.sin((2**i)*directions))
#         directions_encoded.append(torch.cos((2**i)*directions))
    
#     xyz_encoded = torch.cat(xyz_encoded, dim=-1)
#     directions_encoded = torch.cat(directions_encoded, dim=-1)
    return xyz_encoded, directions_encoded


# Raymarching function. Assumes sigma and rgb are shaped as batch of per-ray values.
# Integration times + radiances + densities --> pixel value
def raymarch(t,sigma,rgb,white_bkgd=False):
#     pdb.set_trace()
    t_widths = t[:,1:] - t[:,0:-1]
    transmits = torch.exp(-1*sigma*t_widths)
    alphas = 1.0 - transmits
    
    T = torch.cumprod(transmits + 1e-10, dim=1)
    weights = T*alphas
    renders = torch.sum(weights[:,:,None]*rgb, dim=1)
    
    
    if white_bkgd:
        bkgd = transmits[:,-1]
        renders = renders + bkgd[:,None]
#     renders = torch.clip(renders, min=0.0, max=1.0)

    return renders


def dda(rays, bbox):
    with torch.no_grad():
        # Rearrange bbox into inner and outer (x,y,z) corners
        xs,ys,zs = bbox[0], bbox[1], bbox[2]
        inner = torch.tensor([xs[0],ys[0],zs[0]], device=rays.device)
        outer = torch.tensor([xs[1],ys[1],zs[1]], device=rays.device)

        o,d = rays[:,0:3], rays[:,3:6] # Origins and directions
        t1 = (inner - o) / (d+1e-6)
        t2 = (outer - o) / (d+1e-6)
        t_near = torch.minimum(t1,t2)
        t_far = torch.maximum(t1,t2)
        t_near,_ = torch.max(t_near, dim=1)
        t_far,_ = torch.min(t_far, dim=1)

    #     t_left,_ = torch.max(t_left, dim=1) # Max and min return tuple, which we unpack
    #     t_right,_ = torch.min(t_right, dim=1)
        valid = torch.gt(t_far, t_near) # Boolean tensor mask, t_right>t_left?
    return t_near[valid], t_far[valid], valid


def forward_pass(rays, pixels, mlp, device, num_samples, t_near=2.0, t_far=6.0, L_xyz=2, L_directions=2, white_bkgd=False, use_tensorf=True, pause=False):
    t_near, t_far, valid = dda(rays, mlp.bbox)
    num_rays_orig = rays.shape[0]
    rays = rays[valid]
#     pixels = pixels[valid]
    num_rays_valid = torch.sum(valid==True)
    assert num_rays_valid == rays.shape[0]
#     pdb.set_trace()
#     print(f"{num_rays_valid=}")
#     print(f"{100.0*num_rays_valid/num_rays_orig}% of rays are valid.")
    
    # Sample points along ray between t_near and t_far in integration time
    t = integration_times(num_samples, t_near, t_far, bespoke_t=True)
    t = t.to(device)
    # Convert to xyz+direction in world frame
    xyz,directions = rays_to_points(rays, t)
#     pdb.set_trace()
    # Remove point at t_far for each ray, reshape to batch of sample points
    xyz = xyz[:,:-1,:]
    directions = directions[:,:-1,:]

    if use_tensorf:
        sigma,rgb = mlp(xyz,directions)
        if pause:
            pass
#             pdb.set_trace()
    else:
        # Encode xyz points with Fourier. 
        # Shape is (batch_size, num_samples+1, 2*L*3)
        xyz,directions = encode_all(xyz,directions,L_xyz,L_directions)
        xyz = torch.reshape(xyz, (-1,2*L_xyz*3))
        directions = torch.reshape(directions, (-1,2*L_directions*3))
        # Pass through MLP
        sigma,rgb = mlp(xyz,directions)

    # Reshape to batch of rays and raymarch
    
    sigma = torch.reshape(sigma, (num_rays_valid, num_samples))
    rgb = torch.reshape(rgb, (num_rays_valid, num_samples, -1))
    renders = raymarch(t,sigma,rgb,white_bkgd)
    return renders, valid