import os
from pathlib import Path
import plyfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial.transform import Rotation as R
import argparse

from plot_point_cloud import plot_point_cloud

#! Argparse
parser = argparse.ArgumentParser(description="Point Cloud Pre-Processor")

parser.add_argument(
    "--ply", 
    type=str, 
    help="Path to PLY file"
)

args = parser.parse_args()

#! Read Data
with open(args.ply, "rb") as f:
    ply_data = plyfile.PlyData.read(f)
   
vertex_data = ply_data["vertex"]

num_of_points = len(vertex_data["x"])

x = np.array(vertex_data["x"])
y = np.array(vertex_data["y"])
z = np.array(vertex_data["z"])

r = np.array(vertex_data["f_dc_0"])
g = np.array(vertex_data["f_dc_1"])
b = np.array(vertex_data["f_dc_2"])

points = np.array([x,y,z])
colors = np.array([r,g,b])


#! Initial Rot and Trans
initial_transform_path = Path(os.getenv('WORKSPACE'))/'align_rt.txt'
if initial_transform_path.exists():
    print(f"Loading Transforms from: {initial_transform_path}")
    initial_transforms = np.loadtxt(initial_transform_path)
    euler_angles = initial_transforms[:3]
    trans = initial_transforms[3:]
else:
    print("Using Null Transforms (0,0,0)")
    euler_angles = np.array([0.0, 0.0, 0.0])
    trans = np.array([0.0, 0.0, 0.0])

#! Initial Delta Rot and Trans
delta_trans = 0.1
delta_rot = 1

def euler_to_rotmat(euler_angles):
    rotation = R.from_euler('xyz', np.deg2rad(euler_angles))
    rot_mat = rotation.as_matrix()
    return rot_mat

#! Subsample points
downsample_points = 10000
indices = np.random.choice(num_of_points, size=downsample_points, replace=False)
points = points[:,indices]
colors = colors[:, indices]

#! Normalize DC Colors
cmin = colors.min()
cmax = colors.max()
colors = (colors-cmin)/(cmax-cmin)

#! Plotting
fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(18, 6))
plt.subplots_adjust(bottom=0.3)  # Leave space for the slider
points_copy = points.copy()
rot_mat = euler_to_rotmat(euler_angles)
points_copy = rot_mat@points_copy+trans.reshape(3,1)
plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
fig.canvas.manager.set_window_title("Point Cloud Pre-Processor")


#! Button
move_x_positive_ax = plt.axes([0.2, 0.15, 0.02, 0.05])
move_x_positive_ax.text(-1.5, 0.0, 'Trans:', fontsize=12, color='black')
move_x_positive_button = Button(move_x_positive_ax, '+X', color='lightblue', hovercolor='0.975')
def move_x_positive(event):
    trans[0] += delta_trans
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
move_x_positive_button.on_clicked(move_x_positive)

move_x_negative_ax = plt.axes([0.2, 0.1, 0.02, 0.05])
move_x_negative_button = Button(move_x_negative_ax, '-X', color='lightblue', hovercolor='0.975')
def move_x_negative(event):
    trans[0] -= delta_trans
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
move_x_negative_button.on_clicked(move_x_negative)

move_y_positive_ax = plt.axes([0.22, 0.15, 0.02, 0.05])
move_y_positive_button = Button(move_y_positive_ax, '+Y', color='lightblue', hovercolor='0.975')
def move_y_positive(event):
    trans[1] += delta_trans
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
move_y_positive_button.on_clicked(move_y_positive)

move_y_negative_ax = plt.axes([0.22, 0.1, 0.02, 0.05])
move_y_negative_button = Button(move_y_negative_ax, '-Y', color='lightblue', hovercolor='0.975')
def move_y_negative(event):
    trans[1] -= delta_trans
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
move_y_negative_button.on_clicked(move_y_negative)

move_z_positive_ax = plt.axes([0.24, 0.15, 0.02, 0.05])
move_z_positive_button = Button(move_z_positive_ax, '+Z', color='lightblue', hovercolor='0.975')
def move_z_positive(event):
    trans[2] += delta_trans
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
move_z_positive_button.on_clicked(move_z_positive)

move_z_negative_ax = plt.axes([0.24, 0.1, 0.02, 0.05])
move_z_negative_button = Button(move_z_negative_ax, '-Z', color='lightblue', hovercolor='0.975')
def move_z_negative(event):
    trans[2] -= delta_trans
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
move_z_negative_button.on_clicked(move_z_negative)

#! Buttons for Rotation
rotate_x_positive_ax = plt.axes([0.3, 0.15, 0.02, 0.05])
rotate_x_positive_ax.text(-1.0, 0.0, 'Rot:', fontsize=12, color='black')
rotate_x_positive_button = Button(rotate_x_positive_ax, '+X', color='lightblue', hovercolor='0.975')
def rotate_x_positive(event):
    euler_angles[0] += delta_rot
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
rotate_x_positive_button.on_clicked(rotate_x_positive)

rotate_y_positive_ax = plt.axes([0.32, 0.15, 0.02, 0.05])
rotate_y_positive_button = Button(rotate_y_positive_ax, '+Y', color='lightblue', hovercolor='0.975')
def rotate_y_positive(event):
    euler_angles[1] += delta_rot
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
rotate_y_positive_button.on_clicked(rotate_y_positive)

rotate_z_positive_ax = plt.axes([0.34, 0.15, 0.02, 0.05])
rotate_z_positive_button = Button(rotate_z_positive_ax, '+Z', color='lightblue', hovercolor='0.975')
def rotate_z_positive(event):
    euler_angles[2] += delta_rot
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
rotate_z_positive_button.on_clicked(rotate_z_positive)

rotate_x_negative_ax = plt.axes([0.3, 0.10, 0.02, 0.05])
rotate_x_negative_button = Button(rotate_x_negative_ax, '-X', color='lightblue', hovercolor='0.975')
def rotate_x_negative(event):
    euler_angles[0] -= delta_rot
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
rotate_x_negative_button.on_clicked(rotate_x_negative)

rotate_y_negative_ax = plt.axes([0.32, 0.10, 0.02, 0.05])
rotate_y_negative_button = Button(rotate_y_negative_ax, '-Y', color='lightblue', hovercolor='0.975')
def rotate_y_negative(event):
    euler_angles[1] -= delta_rot
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
rotate_y_negative_button.on_clicked(rotate_y_negative)

rotate_z_negative_ax = plt.axes([0.34, 0.10, 0.02, 0.05])
rotate_z_negative_button = Button(rotate_z_negative_ax, '-Z', color='lightblue', hovercolor='0.975')
def rotate_z_negative(event):
    euler_angles[2] -= delta_rot
    points_copy = points.copy()
    rot_mat = euler_to_rotmat(euler_angles)
    points_copy = rot_mat@points_copy+trans.reshape(3,1)
    plot_point_cloud(points_copy, colors, ax[0], ax[1], ax[2])
    fig.canvas.draw_idle()
rotate_z_negative_button.on_clicked(rotate_z_negative)

#! Button to change delta
delta_trans_positive_ax = plt.axes([0.4, 0.15, 0.02, 0.05])
delta_trans_positive_button = Button(delta_trans_positive_ax, '+ΔT', color='lightblue', hovercolor='0.975')
def delta_trans_positive(event):
    global delta_trans
    delta_trans *= 10
    print(f"Delta Trans: {delta_trans}, Delta Rot: {delta_rot}")
delta_trans_positive_button.on_clicked(delta_trans_positive)

delta_trans_negative_ax = plt.axes([0.4, 0.10, 0.02, 0.05])
delta_trans_negative_button = Button(delta_trans_negative_ax, '-ΔT', color='lightblue', hovercolor='0.975')
def delta_trans_negative(event):
    global delta_trans
    delta_trans /= 10
    print(f"Delta Trans: {delta_trans}, Delta Rot: {delta_rot}")
delta_trans_negative_button.on_clicked(delta_trans_negative)

delta_rot_positive_ax = plt.axes([0.42, 0.15, 0.02, 0.05])
delta_rot_positive_button = Button(delta_rot_positive_ax, '+ΔR', color='lightblue', hovercolor='0.975')
def delta_rot_positive(event):
    global delta_rot
    delta_rot *= 10
    print(f"Delta Trans: {delta_trans}, Delta Rot: {delta_rot}")
delta_rot_positive_button.on_clicked(delta_rot_positive)

delta_rot_negative_ax = plt.axes([0.42, 0.10, 0.02, 0.05])
delta_rot_negative_button = Button(delta_rot_negative_ax, '-ΔR', color='lightblue', hovercolor='0.975')
def delta_rot_negative(event):
    global delta_rot
    delta_rot /= 10
    print(f"Delta Trans: {delta_trans}, Delta Rot: {delta_rot}")
delta_rot_negative_button.on_clicked(delta_rot_negative)

#! Print button
printinfo_ax = plt.axes([0.5, 0.15, 0.05, 0.05])
print_info_button = Button(printinfo_ax, 'Print Rt', color='lightblue', hovercolor='0.975')
def printinfo(event):
    print()
    print(f"Euler: {euler_angles}")
    rot_mat = euler_to_rotmat(euler_angles)
    print(f"Rot:")
    print(rot_mat)
    print(f"Trans: {trans}")
print_info_button.on_clicked(printinfo)

#! Save button
saveinfo_ax = plt.axes([0.58, 0.15, 0.05, 0.05])
save_info_button = Button(saveinfo_ax, 'Save Rt', color='lightblue', hovercolor='0.975')
def saveinfo(event):
    Rt_to_save = np.hstack((euler_angles, trans))
    save_path = Path(os.getenv('WORKSPACE'))/'align_rt.txt' 
    np.savetxt(save_path, Rt_to_save)
    print(f"\nAlignment Rt saved to: {save_path}")
save_info_button.on_clicked(saveinfo)
    

#! Print Info
print(f"Delta Trans: {delta_trans}, Delta Rot: {delta_rot}")

#! Show plot
fig.canvas.manager.set_window_title("Point Cloud Pre-Processor")
plt.show()