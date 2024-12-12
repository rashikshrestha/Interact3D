import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import plyfile
import argparse
import numpy as np
from matplotlib.widgets import Button

#! Argparse
parser = argparse.ArgumentParser(description="Point Cloud Pre-Processor")

parser.add_argument(
    "--ply", 
    type=str, 
    help="Path to PLY file"
)

args = parser.parse_args()

ws = Path(os.getenv('WORKSPACE'))

#! Initial Shift and Scale
initial_transform_path = Path(os.getenv('WORKSPACE'))/'align_shift_scale.txt'
if initial_transform_path.exists():
    print(f"Loading Transforms from: {initial_transform_path}")
    initial_transforms = np.loadtxt(initial_transform_path)
    shift = initial_transforms[:2]
    scale = initial_transforms[2:]
else:
    print("Using Null Transforms")
    shift = np.array([0.0,0.0])
    scale = np.array([1.0])

#! Delta Shift and Scale
delta_shift = 10.0
delta_scale = 0.1

#! Read the image
image_path = ws/'topview.png'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.flip(img, 0)


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

#! Subsample points
# downsample_points = 10000
# indices = np.random.choice(num_of_points, size=downsample_points, replace=False)
# points = points[:,indices]
# colors = colors[:, indices]

from scipy.spatial.transform import Rotation as R
def euler_to_rotmat(euler_angles):
    rotation = R.from_euler('xyz', np.deg2rad(euler_angles))
    rot_mat = rotation.as_matrix()
    return rot_mat



# Rotate
initial_transforms = np.loadtxt(ws/'align_rt.txt')
euler_angles = initial_transforms[:3]
trans = initial_transforms[3:]
rot_mat = euler_to_rotmat(euler_angles)
points = rot_mat@points+trans.reshape(3,1)

#! Alignment
points = points.T
x,y,z = points[:,0], points[:,1], points[:,2]
h, w, _ = img.shape
z = (z-z.min())/(z.max()-z.min()) * w
x = (x-x.min())/(x.max()-x.min()) * h


#! Setup Plot
fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(10,10))
plt.subplots_adjust(bottom=0.2)  # Leave space for the slider

def setup_axes(ax):
    ax.clear()
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
setup_axes(ax)

#! Plot Image
ax.imshow(img)


#! Plot point cloud
cmin = colors.min()
cmax = colors.max()
colors = (colors-cmin)/(cmax-cmin)
x_copy = x.copy()
z_copy = z.copy()
x_copy += shift[0] 
z_copy += shift[1]
x_copy *= scale[0]
z_copy *= scale[0]
ax.scatter(z_copy, x_copy, s=1, alpha=0.2, c=colors.T)



#! Button
move_x_positive_ax = plt.axes([0.2, 0.1, 0.04, 0.04])
move_x_positive_ax.text(-1.7, 0.0, 'Trans:', fontsize=12, color='black')
move_x_positive_button = Button(move_x_positive_ax, '+X', color='lightblue', hovercolor='0.975')
def move_x_positive(event):
    shift[0] += delta_shift
    # Update points and plot
    x_copy = x.copy()
    z_copy = z.copy()
    x_copy += shift[0] 
    z_copy += shift[1]
    x_copy *= scale[0]
    z_copy *= scale[0]
    setup_axes(ax)   
    ax.imshow(img)
    ax.scatter(z_copy, x_copy, s=1, alpha=0.2, c=colors.T)
    fig.canvas.draw_idle()
move_x_positive_button.on_clicked(move_x_positive)

move_x_negative_ax = plt.axes([0.2, 0.06, 0.04, 0.04])
move_x_negative_button = Button(move_x_negative_ax, '-X', color='lightblue', hovercolor='0.975')
def move_x_negative(event):
    shift[0] -= delta_shift
    # Update points and plot
    x_copy = x.copy()
    z_copy = z.copy()
    x_copy += shift[0] 
    z_copy += shift[1]
    x_copy *= scale[0]
    z_copy *= scale[0]
    setup_axes(ax)   
    ax.imshow(img)
    ax.scatter(z_copy, x_copy, s=1, alpha=0.2, c=colors.T)
    fig.canvas.draw_idle()
move_x_negative_button.on_clicked(move_x_negative)

move_z_positive_ax = plt.axes([0.24, 0.1, 0.04, 0.04])
move_z_positive_button = Button(move_z_positive_ax, '+Z', color='lightblue', hovercolor='0.975')
def move_z_positive(event):
    shift[1] += delta_shift
    # Update points and plot
    x_copy = x.copy()
    z_copy = z.copy()
    x_copy += shift[0] 
    z_copy += shift[1]
    x_copy *= scale[0]
    z_copy *= scale[0]
    setup_axes(ax)   
    ax.imshow(img)
    ax.scatter(z_copy, x_copy, s=1, alpha=0.2, c=colors.T)
    fig.canvas.draw_idle()
move_z_positive_button.on_clicked(move_z_positive)

move_z_negative_ax = plt.axes([0.24, 0.06, 0.04, 0.04])
move_z_negative_button = Button(move_z_negative_ax, '-Z', color='lightblue', hovercolor='0.975')
def move_z_negative(event):
    shift[1] -= delta_shift
    # Update points and plot
    x_copy = x.copy()
    z_copy = z.copy()
    x_copy += shift[0] 
    z_copy += shift[1]
    x_copy *= scale[0]
    z_copy *= scale[0]
    setup_axes(ax)   
    ax.imshow(img)
    ax.scatter(z_copy, x_copy, s=1, alpha=0.2, c=colors.T)
    fig.canvas.draw_idle()
move_z_negative_button.on_clicked(move_z_negative)

# + Scale
pscale_ax = plt.axes([0.4, 0.1, 0.04, 0.04])
pscale_ax.text(-1.7, 0.0, 'Scale:', fontsize=12, color='black')
pscale_button = Button(pscale_ax, '+S', color='lightblue', hovercolor='0.975')
def pscale(event):
    scale[0] += delta_scale
    # Update points and plot
    x_copy = x.copy()
    z_copy = z.copy()
    x_copy += shift[0] 
    z_copy += shift[1]
    x_copy *= scale[0]
    z_copy *= scale[0]
    setup_axes(ax)   
    ax.imshow(img)
    ax.scatter(z_copy, x_copy, s=1, alpha=0.2, c=colors.T)
    fig.canvas.draw_idle()
pscale_button.on_clicked(pscale)

nscale_ax = plt.axes([0.4, 0.06, 0.04, 0.04])
nscale_button = Button(nscale_ax, '-S', color='lightblue', hovercolor='0.975')
def nscale(event):
    scale[0] -= delta_scale
    # Update points and plot
    x_copy = x.copy()
    z_copy = z.copy()
    x_copy += shift[0] 
    z_copy += shift[1]
    x_copy *= scale[0]
    z_copy *= scale[0]
    setup_axes(ax)   
    ax.imshow(img)
    ax.scatter(z_copy, x_copy, s=1, alpha=0.2, c=colors.T)
    fig.canvas.draw_idle()
nscale_button.on_clicked(nscale)

#! Button to change delta
delta_trans_positive_ax = plt.axes([0.55, 0.1, 0.04, 0.04])
delta_trans_positive_button = Button(delta_trans_positive_ax, '+ΔT', color='lightblue', hovercolor='0.975')
def delta_trans_positive(event):
    global delta_shift
    delta_shift *= 10
    print(f"Delta Shift: {delta_shift}, Delta Rot: {delta_scale}")
delta_trans_positive_button.on_clicked(delta_trans_positive)

delta_trans_negative_ax = plt.axes([0.55, 0.06, 0.04, 0.04])
delta_trans_negative_button = Button(delta_trans_negative_ax, '-ΔT', color='lightblue', hovercolor='0.975')
def delta_trans_negative(event):
    global delta_shift
    delta_shift /= 10
    print(f"Delta Shift: {delta_shift}, Delta Rot: {delta_scale}")
delta_trans_negative_button.on_clicked(delta_trans_negative)

delta_rot_positive_ax = plt.axes([0.59, 0.1, 0.04, 0.04])
delta_rot_positive_button = Button(delta_rot_positive_ax, '+ΔS', color='lightblue', hovercolor='0.975')
def delta_rot_positive(event):
    global delta_scale
    delta_scale *= 10
    print(f"Delta Shift: {delta_shift}, Delta Rot: {delta_scale}")
delta_rot_positive_button.on_clicked(delta_rot_positive)

delta_rot_negative_ax = plt.axes([0.59, 0.06, 0.04, 0.04])
delta_rot_negative_button = Button(delta_rot_negative_ax, '-ΔS', color='lightblue', hovercolor='0.975')
def delta_rot_negative(event):
    global delta_scale
    delta_scale /= 10
    print(f"Delta Shift: {delta_shift}, Delta Rot: {delta_scale}")
delta_rot_negative_button.on_clicked(delta_rot_negative)

#! Print button
printinfo_ax = plt.axes([0.7, 0.1, 0.06, 0.04])
print_info_button = Button(printinfo_ax, 'Print', color='lightblue', hovercolor='0.975')
def printinfo(event):
    print()
    print(f"Shift: {shift}")
    print(f"Scale: {scale}")
print_info_button.on_clicked(printinfo)

#! Save button
saveinfo_ax = plt.axes([0.7, 0.06, 0.06, 0.04])
save_info_button = Button(saveinfo_ax, 'Save', color='lightblue', hovercolor='0.975')
def saveinfo(event):
    Rt_to_save = np.hstack((shift, scale))
    save_path = Path(os.getenv('WORKSPACE'))/'align_shift_scale.txt' 
    np.savetxt(save_path, Rt_to_save)
    print(f"\nAlignment Shift-Scale saved to: {save_path}")
save_info_button.on_clicked(saveinfo)

fig.canvas.manager.set_window_title("Alignment Tool")
plt.show()
