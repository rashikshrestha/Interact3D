import warp as wp
import warp.sim
import warp.sim.render
from PIL import Image
import numpy as np
from warp.sim import ModelBuilder
import argparse
from pathlib import Path
import os

from interact3d.utils.utils_warp import add_object
from interact3d.utils.utils_conversion import euler_to_quat


def get_body_pose(model, idx=0):
    """
    From Simulation model, get Tran and Rot of a Body of given idx
    
    Returns:
        np.ndarray: (3,) Translation
        np.ndarray: (3,3) Rotation
    """
    pose = model.body_q.numpy()[idx]
    trans = pose[:3]
    quat = pose[3:]
    rot = wp.quat_to_matrix(quat)

    # Convert warp.types.mat33f to numpy array
    # didn't find any direct conversions
    rot_np = [] 
    for row in rot:
        row_np = []
        for column in row:
            row_np.append(column)
        rot_np.append(row_np)
    rot_np = np.array(rot_np)

    return trans, rot_np
   

if __name__=='__main__':
    builder = ModelBuilder()
    
    parser = argparse.ArgumentParser(description="Point Cloud Segmenter")
    parser.add_argument("-x", type=float, default=0.0)
    parser.add_argument("-y", type=float, default=1.0)
    parser.add_argument("-z", type=float, default=0.0)
    parser.add_argument("-r", type=float, default=0.0)
    parser.add_argument("-p", type=float, default=0.0)
    parser.add_argument("-w", type=float, default=0.0)
    args = parser.parse_args()
    
    ws = Path(os.getenv('WORKSPACE'))
    output_dir = ws/'poses'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    #! Add body without rotating
    quat = euler_to_quat(np.array([args.r, args.p, args.y]))
    body = builder.add_body(
        origin=wp.transform((args.x, args.y, args.z), (quat[0], quat[1], quat[2], quat[3])),
        name="object",
        m=1.0,
        armature=0.0
    )
    
    add_object(builder, body, ws/'object_points.txt', downsample=1000)

    # create model
    model = builder.finalize("cuda")
    model.ground = True

    state = model.state()
    control = model.control()  # optional, to support time-varying control inputs
    integrator = wp.sim.SemiImplicitIntegrator()

    renderer = wp.sim.render.SimRendererOpenGL(
                model=model, 
                path="output/",
                scaling=1.0,
                headless=False
            )

    pixel_shape = (1, renderer.screen_height, renderer.screen_width, 3)

    #! for opengl renderer
    # instance_ids = [[0,1]]
    # renderer.setup_tiled_rendering(instance_ids)
    # print(pixel_shape)

    sim_dt = 1/30.0

    for i in range(1000):
        print(i)
        wp.sim.collide(model, state)
        
        state.clear_forces()
        integrator.simulate(model, state, state, dt=0.004, control=control)
        
        trans, rot = get_body_pose(state, idx=0)
        np.savetxt(f"{output_dir}/trans_{i:04d}.txt", trans)
        np.savetxt(f"{output_dir}/rot_{i:04d}.txt", rot)
        
        # Render the frame
        renderer.begin_frame(i * sim_dt)
        renderer.render(state)
        
        #! Save frame
        # pixels = wp.zeros(pixel_shape, dtype=wp.float32)
        # render_mode = 'rgb'
        # renderer.get_pixels(pixels, mode=render_mode)
        # pixels_np = pixels.numpy()*255
        # pixels_np = pixels_np.astype(np.uint8) 
        # pixels_np = pixels_np[0]
        # pil_image = Image.fromarray(pixels_np)
        # pil_image.save(f"frames/{i:08d}.jpg") 
        #! end
        
        renderer.end_frame()