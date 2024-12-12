import torch
from utils.transformation_utils import apply_rotations, apply_cov_rotations
import numpy as np


def transform_obj(init_pos, init_cov, R, T):
    """
    Args:
        init_pos: (N,3) torch.Tensor, Gaussian xyz
        init_cov: (N,6) torch.Tensor, Gaussian Covariances
        R: (3,3) List[torch.Tensor, (3,3)] Rotation Matrix
        T: (3,) torch.Tensor, Translation Vector
    """
    #! Bounding box for goodday
    # Define bounding box: [xmin, ymin, zmin], [xmax, ymax, zmax]
    # bbox_min = torch.tensor([-0.117, -0.05, -0.11])
    # bbox_max = torch.tensor([0.114, 1.0, 0.11])
    
    #! Bounding box for tennis
    bbox_min = torch.tensor([-0.117, -0.00, -0.1])
    bbox_max = torch.tensor([0.114, 1.0, 0.0])

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


def generate_points_on_cube(num_points, cube_size=2):
    points = []
    
    # Half the size of the cube to consider it centered at the origin
    half_size = cube_size / 2
    
    for _ in range(num_points):
        # Randomly choose a face (0-5 for the 6 faces)
        face = np.random.randint(6)
        
        if face == 0:  # Front face (z = half_size)
            x = np.random.uniform(-half_size, half_size)
            y = np.random.uniform(-half_size, half_size)
            z = half_size
        elif face == 1:  # Back face (z = -half_size)
            x = np.random.uniform(-half_size, half_size)
            y = np.random.uniform(-half_size, half_size)
            z = -half_size
        elif face == 2:  # Top face (y = half_size)
            x = np.random.uniform(-half_size, half_size)
            y = half_size
            z = np.random.uniform(-half_size, half_size)
        elif face == 3:  # Bottom face (y = -half_size)
            x = np.random.uniform(-half_size, half_size)
            y = -half_size
            z = np.random.uniform(-half_size, half_size)
        elif face == 4:  # Left face (x = -half_size)
            x = -half_size
            y = np.random.uniform(-half_size, half_size)
            z = np.random.uniform(-half_size, half_size)
        else:  # Right face (x = half_size)
            x = half_size
            y = np.random.uniform(-half_size, half_size)
            z = np.random.uniform(-half_size, half_size)

        points.append([x, y, z])
    
    return np.array(points)