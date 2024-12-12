from PIL import Image
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

from interact3d.utils.utils_plot import plot_bounding_boxes
from interact3d.models.sam import SAM
from interact3d.models.grounding_dino import GroundingDINO
from interact3d.utils.utils_io import read_ply
from interact3d.utils.utils_conversion import euler_to_rotmat

#! Argparse
parser = argparse.ArgumentParser(description="Point Cloud Pre-Processor")
args = parser.parse_args()

#! Params
device = 'cuda' 
ws = Path(os.getenv('WORKSPACE'))
text_prompt = 'green box.'

#! Read ply
points, colors = read_ply(ws/'gaussians.ply')
initial_transform_path = ws/'align_rt.txt'
initial_transforms = np.loadtxt(initial_transform_path)
euler_angles = initial_transforms[:3]
trans = initial_transforms[3:]
# Transform points
rot_mat = euler_to_rotmat(euler_angles)
points = rot_mat@points+trans.reshape(3,1)

x,y,z = points

#! Models
gdino = GroundingDINO(device, text_prompt, box_th=0.3, text_th=0.3)
sam = SAM(device)


#! Read image
img_pil = Image.open(ws/'topview.png') # PIL (RGB)
img_cv = np.array(img_pil) # OpenCV RGB
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # OpenCV BGR
h,w,_ = img_cv.shape

#! Detection
bb_dino = gdino.detect(img_cv)
bb_dino = [[342, 302, 471, 432]]
img_det = plot_bounding_boxes(img_cv.copy(), bb_dino)

cv2.imwrite(str(ws/'detection.png'), img_det)

mask = sam.get_segmentation_mask(img_pil, bb_dino)
cv2.imwrite(str(ws/'mask.png'), mask) 

#! Image coordinate to Point Cloud Coordinate
# Read shift and scale
align_sh_sc = np.loadtxt(ws/'align_shift_scale.txt')
shift, scale = align_sh_sc[:2], align_sh_sc[-1]

# Detection Results
xmin, ymin, xmax, ymax = bb_dino[0]  # x = w = z, y = h = x

# 1. Undo scale
xmin /= scale
ymin /= scale
xmax /= scale
ymax /= scale

# 2. Undo shift
xmin -= shift[1]
xmax -= shift[1]
ymin -= shift[0]
ymax -= shift[0]

# 3. Undo Scale
xmin /= w
xmax /= w
ymin /= h
ymax /= h

# 4. Undo normalization
xmin = (xmin)*(z.max()-z.min())
xmax = (xmax)*(z.max()-z.min())
ymin = (ymin)*(x.max()-x.min())
ymax = (ymax)*(x.max()-x.min())

xmin += z.min()
xmax += z.min()
ymin += x.min()
ymax += x.min()


pc_bb = np.array([xmin, ymin, xmax, ymax])

np.savetxt(ws/'bbox.txt', pc_bb)