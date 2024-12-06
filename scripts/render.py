import torch
torch.set_printoptions(sci_mode=False, precision=6)
import cv2
import numpy as np
from tqdm import tqdm

#! Interact3d Imports
from interact3d.utils.utils_model import load_gaussian_model_from_ply
from interact3d.utils.utils_model import get_camera_view

#! Physgaussian Imports
from utils.decode_param import decode_param_json
from utils.render_utils import load_params_from_gs, initialize_resterize
from utils.transformation_utils import get_center_view_worldspace_and_observant_coordinate
from utils.transformation_utils import transform2origin, generate_rotation_matrices
from utils.transformation_utils import apply_rotations, apply_cov_rotations

#! Parameters
ply_path = '/home/rashik_shrestha/ws/Interact3D/temp/splat_cropped.ply'
output_path = '/home/rashik_shrestha/ws/Interact3D/temp/frames'
config_path = '/home/rashik_shrestha/ws/Interact3D/temp/test_config.json'
cameras_path = '/home/rashik_shrestha/ws/Interact3D/temp/cameras.json'

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


#! Config (remove this later in original code)
print("Loading scene config...")
(
    material_params,
    bc_params,
    time_params,
    preprocessing_params,
    camera_params,
) = decode_param_json(config_path)

#! Gaussian Model
gaussians = load_gaussian_model_from_ply(ply_path)
pipeline = PipelineParamsNoparse() # has 3 params: convert_SH_py, compute_Cov_py, debug
pipeline.compute_cov3D_python = True

background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")


#! Extract data from Gaussians
print("Extracting Gaussians data")
params = load_params_from_gs(gaussians, pipeline)

init_pos = params["pos"]
init_screen_points = params["screen_points"]
init_cov = params["cov3D_precomp"]
init_opacity = params["opacity"]
init_shs = params["shs"]
print(f"Total Gaussians: {init_pos.shape[0]}")
# print(init_pos.shape, init_screen_points.shape, init_shs.shape, init_cov.shape, init_opacity.shape)

#! Filter out gaussians less than given opacity threshold
mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
init_pos = init_pos[mask, :]
init_screen_points = init_screen_points[mask, :]
init_cov = init_cov[mask, :]
init_opacity = init_opacity[mask, :]
init_shs = init_shs[mask, :]
print(f"Gaussians after opacity filter: {init_pos.shape[0]}")

#! Rotate and Tranlate the gaussians here
gaussian_rotation_matrices = generate_rotation_matrices(
    torch.tensor([82.3, -39.6, 180.3]),
    [0,1,2],
)
print("rotation matrices")
print(gaussian_rotation_matrices)
# print(gaussian_rotation_matrices.shape, gaussian_rotation_matrices.device)
init_pos = apply_rotations(init_pos, gaussian_rotation_matrices)
init_cov = apply_cov_rotations(init_cov, gaussian_rotation_matrices)
trans_vec = torch.tensor([-0.088, 0.45, -0.067]).to('cuda')
init_pos += trans_vec

#! Get Camera
# rorate and translate object
rotation_matrices = generate_rotation_matrices(
    torch.tensor(preprocessing_params["rotation_degree"]),
    preprocessing_params["rotation_axis"],
)
rotated_pos = apply_rotations(init_pos, rotation_matrices)
    
transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    
mpm_space_viewpoint_center = (
    torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
)
mpm_space_vertical_upward_axis = (
    torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
    .reshape((1, 3))
    .cuda()
)
(
    viewpoint_center_worldspace,
    observant_coordinates,
) = get_center_view_worldspace_and_observant_coordinate(
    mpm_space_viewpoint_center,
    mpm_space_vertical_upward_axis,
    rotation_matrices,
    scale_origin,
    original_mean_pos,
)
    
frame_number = 0 # camera will be rotated on the basis of frame number
camera_view = get_camera_view(
    cameras_path,
    default_camera_index=camera_params["default_camera_index"],
    center_view_world_space=viewpoint_center_worldspace,
    observant_coordinates=observant_coordinates,
    show_hint=camera_params["show_hint"],
    init_azimuthm=camera_params["init_azimuthm"],
    init_elevation=camera_params["init_elevation"],
    init_radius=camera_params["init_radius"],
    move_camera=camera_params["move_camera"],
    current_frame=frame_number,
    delta_a=camera_params["delta_a"],
    delta_e=camera_params["delta_e"],
    delta_r=camera_params["delta_r"],
)

#! Object Rotate and Translate
def transform_obj(init_pos, init_cov, R, T):
    """
    Args:
        init_pos: (N,3) torch.Tensor
        init_cov: (N,6) torch.Tensor
        R: (3,3) List[torch.Tensor, (3,3)]
        T: (3,) torch.Tensor
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

for i in tqdm(range(1000)):
    print(f"/home/rashik_shrestha/ws/PhysGaussian/box_poses/rot_{i:04d}.txt")

    # print('just before rasterization:') 
    box_R_np = np.loadtxt(f"/home/rashik_shrestha/ws/PhysGaussian/box_poses/rot_{i:04d}.txt")
    box_T_np = np.loadtxt(f"/home/rashik_shrestha/ws/PhysGaussian/box_poses/trans_{i:04d}.txt")


    box_R = [torch.as_tensor(box_R_np, dtype=torch.float32, device='cuda')]
    box_T = torch.as_tensor(box_T_np, dtype=torch.float32, device='cuda')


    #! Rasterize
    rasterize = initialize_resterize(
        camera_view, gaussians, pipeline, background
    )

    init_pos_new, init_cov_new = transform_obj(torch.clone(init_pos), torch.clone(init_cov), box_R, box_T)

    rendering, raddi = rasterize(
        means3D=init_pos_new,
        means2D=init_screen_points,
        shs=init_shs,
        colors_precomp=None,
        opacities=init_opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=init_cov_new,
    )

    cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    print(cv2_img.shape)

    cv2.imwrite(f"{output_path}/rendered_{i:04d}.png", cv2_img*255)