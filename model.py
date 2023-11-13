import torch
import torch.nn as nn
import torch.optim as optim
# import torch.cuda.amp as amp
import torch.nn.functional as F

from rays import encode
import math
import pdb

# TODO: b vectors, outerproducts, linear interpolation, L1 regularization (based on number of components)?,
# coarse-to-fine, different learning rates for tensor factors vs MLP
class Tensorf(nn.Module): # 16,48 *3 = 48,144
    def __init__(self, channels=128, R_s=48, R_c=144, P=27, L=2, N_init=128, N_final=500, bbox=None, w=1e-5):
        super().__init__()
        self.R_s = R_s # Number of components
        self.R_c = R_c
        self.P = P
        self.channels = channels
        self.L = L
        self.N_init = N_init
        self.N_final = N_final
        self.N = N_init # Number of voxels along each spatial dimension
        self.bbox = bbox # Bounding box around object
        self.w = w # L1 loss weight
        self.sigma_bias = -5.0
        
        # MLP
        self.S = nn.Sequential(
            # Feature vector plus direction encoded with L frequencies, both sine and cosine
            nn.Linear((self.P+3)*L*2, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, 3),
            nn.Sigmoid()
        )
        # Voxel corners by dimension
        voxel = [torch.linspace(bbox[i][0],bbox[i][1],steps=self.N, requires_grad=False) for i in range(3)]
        voxel = torch.stack(voxel, dim=0)
        self.register_buffer(f"voxel", voxel)
        # Factorized density tensor components 
        self.sigma = nn.Parameter(0.1*torch.randn(size=(3, R_s, self.N)))
        # Factorized color radiance components
        self.feature = nn.Parameter(0.1*torch.randn(size=(3, R_c, self.N)))
        self.B = nn.Parameter(torch.randn(size=(R_c,P)))
    
    def L1_penalty(self):
#         L1_loss = torch.tensor(0.0)
#         for sigma_param in self.sigma:
        absval = 0
        numel = 0
        params_list = [self.sigma, self.feature, self.B]
        for param in params_list:
            absval += torch.sum(torch.abs(param))
            numel += torch.numel(param)
        L1_loss = self.w * (absval / numel)
        return L1_loss
    
    def upsample(self, steps, total_upscale_steps, total_steps, num_epochs, epoch, optim_old=None):
        with torch.no_grad():
            # Interpolate number of voxels in log-space
            logNi_3 = math.log(self.N_init**3)
            logNf_3 = math.log(self.N_final**3)
            lerp_weight = steps / total_upscale_steps
            logNc_3 = logNi_3 + lerp_weight*(logNf_3 - logNi_3)
            Nc = int((math.exp(logNc_3))**(1.0/3)) # take exponent, cube root, round to int
            self.N = Nc
            # Upsample tensor factors by using new N value
            voxel = [torch.linspace(self.bbox[i][0],self.bbox[i][1],steps=self.N, requires_grad=False) for i in range(3)]
            voxel = torch.stack(voxel, dim=0)
            self.register_buffer(f"voxel", voxel)
#             voxel = F.interpolate(self.voxel, size=Nc, mode='linear')
#             self.register_buffer(f"voxel", voxel)
            self.sigma = nn.Parameter(F.interpolate(self.sigma, size=Nc, mode='linear'))
            self.feature = nn.Parameter(F.interpolate(self.feature, size=Nc, mode='linear'))
            
            # New optimizer w/ upsampled tensor factors
            optim_new = optim.Adam(
                [
                    {'params':self.sigma, 'lr':0.02},
                    {'params':self.feature, 'lr':0.02},
                    {'params':self.B, 'lr':0.02},
                    {'params':self.S.parameters(), 'lr':0.001},
                ],
                lr=1e-3
            )
            # New scheduler from new optimizer
            scheduler_new = optim.lr_scheduler.ExponentialLR(
                optim_new, gamma=0.1**(1/total_steps), last_epoch=-1
            )
            # Progress scheduler to current epoch
            for i in range(steps):
                scheduler_new.step()
                
            print(f"Upsampled! {Nc=}.")
#         pdb.set_trace()
        return optim_new, scheduler_new
    
    # Lookup sigma density from location
    def get_density(self, xyz, vox, inds):
        # (batch, R^3 value) snaps via searchsorted to marked-out axes. Using resulting indices, index into
        # density lookup table, get densities. Then interpolate using original xyz and marked-out xyz values
        # using searchsorted indices.

        vox_left, vox_right = vox
        inds_left, inds_right = inds
        # Linear interpolate between voxel points:
        # (3, R_s, N), we're indexing along dim=2
        s_left = torch.take_along_dim(self.sigma, inds_left[:,None,:], 2)
        s_right = torch.take_along_dim(self.sigma, inds_right[:,None,:], 2)
        lerp_weights = (xyz - vox_left)/(vox_right - vox_left + 1e-6)
        s_lerped = s_left + (s_right-s_left)*lerp_weights[:,None,:]

        # Permute for hopefully better performance
        s_lerped = torch.permute(s_lerped, (2,1,0)).contiguous() # (3, R, batch) --> (batch, R, 3)
        assert s_lerped.is_contiguous()
        sigmas = torch.prod(s_lerped, dim=2) # outer product
        sigmas = torch.sum(sigmas, dim=1) # sum rank-1 components
#         sigmas = sigmas.T.contiguous() # transpose to put batch dim first
        return sigmas


    def get_features(self, xyz, vox, inds):
        features = []
        vox_left, vox_right = vox
        inds_left, inds_right = inds
        # Lerp
        f_left = torch.take_along_dim(self.feature, inds_left[:,None,:], 2)
        f_right = torch.take_along_dim(self.feature, inds_right[:,None,:], 2)
        lerp_weights = (xyz - vox_left)/(vox_right - vox_left + 1e-6)
        f_lerped = f_left + (f_right-f_left)*lerp_weights[:,None,:]
        # Permute for performance, hopefully
#         pdb.set_trace()
        f_lerped = torch.permute(f_lerped, (2,1,0)).contiguous()
        features = torch.prod(f_lerped, dim=2)
        features = features @ self.B # matmul to get feature vectors / sum components 
#         features = features.T.contiguous() # transpose to put batch dim first
        return features

    
    def snap_to_voxels(self, xyz):    
        # indices of nearest voxel corners
        voxel = getattr(self, f"voxel")
        inds = torch.searchsorted(voxel, xyz)
        
        # clip indices 
        inds_left = torch.clamp(inds-1, 0, self.N-1)
        inds_right = torch.clamp(inds, 0, self.N-1)
        
        # Get left and right indices per dimension
        vox_left = torch.take_along_dim(voxel, inds_left, 1) #torch.gather(voxel, 1, inds_left)
        vox_right = torch.take_along_dim(voxel, inds_right, 1) #torch.gather(voxel, 1, inds_right)

        return inds, (vox_left,vox_right), (inds_left, inds_right)
        
    def forward(self, xyz, directions):
        xyz = torch.reshape(xyz, (-1,3)).T.contiguous() # Shape == (3, -1)
        directions = torch.reshape(directions, (-1,3))
        inds, vox, inds = self.snap_to_voxels(xyz)
        # density
        sigmas = self.get_density(xyz, vox, inds)
        sigmas = F.softplus(sigmas + self.sigma_bias, beta=1.0)
        
        # directional radiance
        features = self.get_features(xyz, vox, inds)
        features = encode(features, self.L)
        directions = encode(directions, self.L)
#         print(f"{torch.cat((features, directions), dim=1).shape=}")
        rgb = self.S(torch.cat((features, directions), dim=1))
        
        if torch.sum(torch.isnan(sigmas)==True) > 0:
            print(f"Nan in sigmas")
#             pdb.set_trace
        if torch.sum(torch.isnan(rgb)==True) > 0:
            print(f"Nan in rgb")
        return sigmas, rgb



#######################################################################################################
# Minimal vanilla-Nerf style model

# Basic set of layers, repeat to increase model depth arbitrarily
class fc_layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
        )
    def forward(self, input):
        return self.layer(input)

class Nerf(nn.Module):
    def __init__(self, L_xyz=10, L_directions=4, channels=128, num_layers=4):
        super().__init__()
        self.xyz_width = 2*L_xyz*3
        self.directions_width = 2*L_directions*3
        self.channels = channels
        self.num_layers = num_layers

        
        self.net = nn.Sequential(
            nn.Linear(self.xyz_width + self.directions_width, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
        )
        self.density = nn.Sequential(
            nn.Linear(channels, 1),
            nn.Softplus(beta=1),
        )
        self.color = nn.Sequential(
            nn.Linear(channels, 3),
            nn.Sigmoid()
        )


    def forward(self, xyz, directions):
        input_vector = torch.cat([xyz,directions], dim=1)
        feature = self.net(input_vector)
        s = self.density(feature)
        rgb = self.color(feature)
        return s, rgb
