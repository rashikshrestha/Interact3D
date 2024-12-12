import numpy as np

from interact3d.utils.utils_geometry import generate_points_on_cube


def add_object(builder, body, points_path=None, downsample=None):
    if points_path is None:
        points = generate_points_on_cube(num_points=1000, cube_size=0.5)
    else:
        points = np.loadtxt(points_path)
       
    if downsample is not None: 
        indices = np.random.choice(points.shape[0], size=downsample, replace=False)
        points = points[indices]
    
    point_size = 0.02
    
    for point in points:
        builder.add_shape_sphere(
            body,
            pos = point,
            radius = point_size,
            thickness=0.01,
            density = 100,
            kd = 1.0e3,
        )