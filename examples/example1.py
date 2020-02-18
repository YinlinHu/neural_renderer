"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import os
import argparse

import torch
import numpy as np
import tqdm
import imageio
import trimesh
import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'test.ply'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example1.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    # camera_distance = 2.732
    # elevation = 30
    # texture_size = 1

    # load .obj
    # vertices, faces = nr.load_obj(args.filename_input)
    # vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    # faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    # textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # load mesh by trimesh
    mesh = trimesh.load(args.filename_input)
    # 
    # [batch_size=1, num_vertices, XYZ]
    vertices = torch.from_numpy(mesh.vertices).float().cuda().unsqueeze(0)
    # 
    # [batch_size=1, num_faces, 3]
    faces = torch.from_numpy(mesh.faces).int().cuda().unsqueeze(0)
    # 
    # [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = (torch.from_numpy(mesh.visual.face_colors[:,:3])/255.0).cuda().view(1,-1,1,1,1,3)

    # create renderer
    renderer = nr.Renderer(camera_mode='projection')
    # K: batch_size * 3 * 3 intrinsic camera matrix
    # R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters

    renderer.orig_size = 640
    renderer.image_size = 320
    renderer.background_color = [0.5,0.5,0.5] # gray background
    renderer.K = torch.FloatTensor([[1.066778e+03,0.000000e+00,3.129869e+02]
                                    ,[0.000000e+00,1.067487e+03,2.413109e+02]
                                    ,[0.000000e+00,0.000000e+00,1.000000e+00]]).cuda()
    # renderer.K[1][1] = -renderer.K[1][1]
    # renderer.K[1][2] = -renderer.K[1][2] + renderer.orig_size
    renderer.K = renderer.K.view(1,3,3)
    renderer.R = torch.FloatTensor([[-0.98050106,-0.00619752,0.19641621]
                                    ,[-0.05675691,-0.94797665,-0.31323951]
                                    ,[0.18813927,-0.31827962, 0.92914033]]).cuda().view(1,3,3)
    renderer.t = torch.FloatTensor([[ 0.202833]
                                    ,[-0.032759]
                                    ,[ 0.881898]]).cuda().view(1,1,3)


    # draw object
    loop = tqdm.tqdm(range(0, 8, 4))
    writer = imageio.get_writer(args.filename_output, mode='I')
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        # renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images, _, _ = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        writer.append_data((255*image).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()
