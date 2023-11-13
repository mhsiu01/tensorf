import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image as Img
import cv2

from tqdm import tqdm
from rays import forward_pass

# Converts list of pixels to PIL image
def pixels_to_img(render_img, D):
#     render_img = torch.cat(render_img).cpu()
    render_img = np.asarray(render_img.cpu())
    render_img = np.reshape(render_img, (D,D,3))

    render_img = np.clip(render_img, 0.0, 1.0)
    render_img = Img.fromarray(np.uint8(render_img*255))
    return render_img

def render(mlp, loader, num_samples, L_xyz, L_directions, white_bkgd, D, use_tensorf):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        valid_render_pixels = []
        valid_mask = []
        true_pixels = []
        for iteration, (rays, pixels) in enumerate(tqdm(loader, ascii=True)):
            rays = rays.to(device, non_blocking=True)
            pixels = pixels.to(device, non_blocking=True)
#             with amp.autocast(enabled=False):
            renders, valid = forward_pass(rays, pixels, mlp, device, num_samples*2)
            loss = F.mse_loss(renders, pixels[valid])
            valid_render_pixels.append(renders)
            valid_mask.append(valid)
            true_pixels.append(pixels)
        # For every chunk of pixels belonging to one image,
        # convert to displayable image
        
        # List of minibatch values --> one big batch tensor
        valid_mask = torch.cat(valid_mask, dim=0)
        true_pixels = torch.cat(true_pixels, dim=0)
        valid_render_pixels = torch.cat(valid_render_pixels, dim=0)
        # Reinsert blank pixels for invalid rays
        render_pixels = torch.zeros(len(true_pixels),3, device=device)
        render_pixels[valid_mask] = valid_render_pixels
        
        
#         render_imgs = [render_pixels[i:i+D*D] for i in range(0,len(render_pixels),D*D)]
#         true_imgs = [true_pixels[i:i+D*D] for i in range(0,len(true_pixels),D*D)]
        results = []
        for i in range(0,len(render_pixels),D*D):
            render_img = pixels_to_img(render_pixels[i:i+D*D], D)
            display(render_img)
            true_img = pixels_to_img(true_pixels[i:i+D*D], D)
            display(true_img)
            psnr = get_psnr(render_img, true_img)
            results.append((render_img, true_img, psnr))
    return results

def get_psnr(img1, img2):
    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    psnr = cv2.PSNR(img1_cv, img2_cv)
    mse = np.mean(np.square(np.array(img1)/255 - np.array(img2)/255))
    psnr_ = 20.0*np.log10(1.0) - 10.0*np.log10(mse)
    print(f"{psnr_=}   {psnr=}")

    return psnr