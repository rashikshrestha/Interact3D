import os
import json
import torch
import numpy as np


#! physgaussian imports
from utils.camera_view_utils import get_current_radius_azimuth_and_elevation, get_camera_position_and_rotation, focal2fov

#! Gaussian Splatting imports
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera as GSCamera


def load_gaussian_model_from_ply(ply_path, sh_degree=3):
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    return gaussians

def apply_opacity_filter(threshold, means3d, means2d, shs, opacities, covs3D):
    mask = opacities[:, 0] > threshold
    means3d = means3d[mask, :]
    means2d = means2d[mask, :]
    covs3D = covs3D[mask, :]
    opacities = opacities[mask, :]
    shs = shs[mask, :]
    return means3d, means2d, shs, opacities, covs3D

def get_camera_view(
    cam_path,
    default_camera_index=0,
    center_view_world_space=None,
    observant_coordinates=None,
    show_hint=False,
    init_azimuthm=None,
    init_elevation=None,
    init_radius=None,
    move_camera=False,
    current_frame=0,
    delta_a=0,
    delta_e=0,
    delta_r=0,
):
    """Load one of the default cameras for the scene."""
    with open(cam_path) as f:
        data = json.load(f)

        if show_hint:
            if default_camera_index < 0:
                default_camera_index = 0
            r, a, e = get_current_radius_azimuth_and_elevation(
                data[default_camera_index]["position"],
                center_view_world_space,
                observant_coordinates,
            )
            print("Default camera ", default_camera_index, " has")
            print("azimuth:    ", a)
            print("elevation:  ", e)
            print("radius:     ", r)
            print("Now exit program and set your own input!")
            exit()

        if default_camera_index > -1:
            raw_camera = data[default_camera_index]

        else:
            raw_camera = data[0]  # get data to be modified

            assert init_azimuthm is not None
            assert init_elevation is not None
            assert init_radius is not None

            if move_camera:
                assert delta_a is not None
                assert delta_e is not None
                assert delta_r is not None
                position, R = get_camera_position_and_rotation(
                    init_azimuthm + current_frame * delta_a,
                    init_elevation + current_frame * delta_e,
                    init_radius + current_frame * delta_r,
                    center_view_world_space,
                    observant_coordinates,
                )
            else:
                position, R = get_camera_position_and_rotation(
                    init_azimuthm,
                    init_elevation,
                    init_radius,
                    center_view_world_space,
                    observant_coordinates,
                )
            raw_camera["rotation"] = R.tolist()
            raw_camera["position"] = position.tolist()

        tmp = np.zeros((4, 4))
        tmp[:3, :3] = raw_camera["rotation"]
        tmp[:3, 3] = raw_camera["position"]
        tmp[3, 3] = 1
        C2W = np.linalg.inv(tmp)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]

        width = raw_camera["width"]
        height = raw_camera["height"]
        fovx = focal2fov(raw_camera["fx"], width)
        fovy = focal2fov(raw_camera["fy"], height)

        return GSCamera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros((3, height, width)),  # fake
            gt_alpha_mask=None,
            image_name="fake",
            uid=0,
        )