from scipy.spatial.transform import Rotation as R
import numpy as np


def euler_to_rotmat(euler_angles):
    rotation = R.from_euler('xyz', np.deg2rad(euler_angles))
    rot_mat = rotation.as_matrix()
    return rot_mat


def euler_to_quat(euler_angles):
    rotation = R.from_euler('xyz', np.deg2rad(euler_angles))
    quat = rotation.as_quat()
    return quat
    
