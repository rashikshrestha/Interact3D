import numpy as np
import torch
import cv2
import os
import plyfile

def get_obj_pose(path, idx, device):
    box_R_np = np.loadtxt(f"{path}/rot_{idx:04d}.txt")
    box_T_np = np.loadtxt(f"{path}/trans_{idx:04d}.txt")
    box_R = [torch.as_tensor(box_R_np, dtype=torch.float32, device=device)]
    box_T = torch.as_tensor(box_T_np, dtype=torch.float32, device=device)
    return box_R, box_T


def save_torch_image(img_torch, path):
    cv2_img = img_torch.permute(1, 2, 0).detach().cpu().numpy()
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)*255
    cv2.imwrite(path, cv2_img)
   
def render_video(path, fps=120):
    os.system(
        f"ffmpeg -framerate {fps} -i {path}/rendered_%04d.png -y -c:v libx264 -pix_fmt yuv420p {path}/output.mp4"
    )
    

def read_ply(ply_file):
    with open(ply_file, "rb") as f:
        ply_data = plyfile.PlyData.read(f)
   
    vertex_data = ply_data["vertex"]

    x = np.array(vertex_data["x"])
    y = np.array(vertex_data["y"])
    z = np.array(vertex_data["z"])

    r = np.array(vertex_data["f_dc_0"])
    g = np.array(vertex_data["f_dc_1"])
    b = np.array(vertex_data["f_dc_2"])

    points = np.array([x,y,z])
    colors = np.array([r,g,b])
    
    return points, colors