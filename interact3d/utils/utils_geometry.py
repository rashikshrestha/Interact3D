import torch
from utils.transformation_utils import apply_rotations, apply_cov_rotations


def transform_obj(init_pos, init_cov, R, T):
    """
    Args:
        init_pos: (N,3) torch.Tensor, Gaussian xyz
        init_cov: (N,6) torch.Tensor, Gaussian Covariances
        R: (3,3) List[torch.Tensor, (3,3)] Rotation Matrix
        T: (3,) torch.Tensor, Translation Vector
    """
    # Define bounding box: [xmin, ymin, zmin], [xmax, ymax, zmax]
    bbox_min = torch.tensor([-0.117, -0.05, -0.11])
    bbox_max = torch.tensor([0.114, 1.0, 0.11])

    # Create a mask to select points within the bounding box
    mask = (init_pos[:, 0] >= bbox_min[0]) & (init_pos[:, 0] <= bbox_max[0]) & \
           (init_pos[:, 1] >= bbox_min[1]) & (init_pos[:, 1] <= bbox_max[1]) & \
           (init_pos[:, 2] >= bbox_min[2]) & (init_pos[:, 2] <= bbox_max[2])
    
    new_pos = apply_rotations(init_pos[mask], R)
    new_pos += T
    init_pos[mask] = new_pos
    new_cov = apply_cov_rotations(init_cov[mask], R)
    init_cov[mask] = new_cov

    return init_pos, init_cov