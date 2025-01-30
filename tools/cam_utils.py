import torch


def normalize_cam(cam_mask):
    """Normalize the CAM mask."""
    for i in range(cam_mask.size(0)):
        channel = cam_mask[i]
        min_val = torch.min(channel)
        max_val = torch.max(channel)
        cam_mask[i] = (channel - min_val) / (max_val - min_val + 1e-8)
    
    return cam_mask


def flip_cam(cam_list):
    """Flip cam with scales in the given cam_list."""
    for i, cam_scale in enumerate(cam_list):
        group1, group2 = cam_scale[0], cam_scale[1]
        group2_flipped = torch.flip(group2, dims=[2])
        cam_list[i] = torch.stack([group1, group2_flipped])  
    cam_list = [torch.sum(cam, dim=0) for cam in cam_list]
    return cam_list

