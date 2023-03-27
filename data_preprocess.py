import pyexr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import re

def RGB_to_YCoCg(rgb):
    return np.dot(rgb, np.array([[1,  2, -1],
                                 [2,  0,  2],
                                 [1, -2, -1]]))

def YCoCg_to_RGB(YCoCg):
    return np.dot(YCoCg, np.array([[ 0.25, 0.25,  0.25],
                                   [ 0.25,    0, -0.25],
                                   [-0.25, 0.25, -0.25]]))

def process_data():
    # the processing time is slow because it uses a single thread
    scene_names = ["sponza", "classroom", "living-room", "san-miguel", "sponza-glossy", "sponza-moving-light"]

    POSITION_LIMIT_SQUARED = {"sponza":0.001600, "classroom":0.010000, "living-room":0.000900, "san-miguel":0.000900, "sponza-glossy":0.001600, "sponza-moving-light": 0.001600}
    NORMAL_LIMIT_SQUARED = {"sponza":0.250000, "classroom":1.000000, "living-room":0.040000, "san-miguel":1.000000, "sponza-glossy":0.250000, "sponza-moving-light": 0.250000}

    for scene_name in scene_names:

        os.makedirs(os.path.join("dataset", scene_name, "acc_colors"), exist_ok=True)

        BLEND_ALPHA = 0.2
        camera_matrices = np.zeros((60, 4, 4))
        with open(os.path.join("dataset", "cameras", scene_name+".h"), "r") as file:
            camera_idx = 0
            row_idx = 0
            for line in file.readlines():
                floats = re.findall(r'-?\d+.\d+', line)
                if len(floats) > 0:
                    camera_matrices[camera_idx, row_idx] = np.array(floats, dtype=np.float32)
                    row_idx += 1
                if row_idx == 4:
                    camera_idx += 1
                    row_idx = 0


        world_positions = []
        normals = []
        colors = []
        colors_post_accumulated = []
        in_prev_frame_pixel = []
        for i in tqdm(range(60)):
            world_positions.append(pyexr.read(os.path.join("dataset", scene_name, "inputs", "world_position"+str(i)+".exr")))
            normals.append(pyexr.read(os.path.join("dataset", scene_name, "inputs", "shading_normal"+str(i)+".exr")))
            colors.append(pyexr.read(os.path.join("dataset", scene_name, "inputs", "color"+str(i)+".exr")))
            
            
        # H, W = colors[0].shape[:2]
        H, W = world_positions[0].shape[:2]
        H_window, W_window = world_positions[0].shape[:2]

        current_spp = np.zeros((H, W))
        previous_spp = np.zeros((H, W))
        colors_post_accumulated = np.zeros((60, H, W, 3))
        in_prev_frame_pixel = np.zeros((60, H, W, 2))

        world_position = np.array([0., 0., 0., 1.])
        # Step1: post accumulation
        for i in tqdm(range(60)):
            for h_i in range(H):
                for w_i in range(W):
                    world_position[:3] = world_positions[i][h_i, w_i]
                    normal = normals[i][h_i, w_i]
                    current_color = colors[i][h_i, w_i]
                    
                    blend_alpha = 1.0
                    previous_color = np.array([0.0, 0.0, 0.0])
                    sample_spp = 0.0
                    
                    if i > 0:
                        # compute previous frame uv
                        prev_frame_uv = np.dot(world_position, camera_matrices[i-1])[(0, 1, 3),]
                        prev_frame_uv = prev_frame_uv[:2] / prev_frame_uv[2]
                        prev_frame_uv = (prev_frame_uv + 1) / 2

                        prev_frame_pixel_f = prev_frame_uv * np.array([W_window, H_window])
                        prev_frame_pixel_f -= np.array([0.5, 0.5]) # pixel offset
                        in_prev_frame_pixel[i][h_i, w_i] = prev_frame_pixel_f
                        prev_frame_pixel = np.floor(prev_frame_pixel_f).astype(np.int)

                        # These are needed for bilinear sampling
                        offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
                        prev_pixel_fract = prev_frame_pixel_f - prev_frame_pixel
                        one_minus_prev_pixel_fract = 1.0 - prev_pixel_fract
                        weights = np.array([one_minus_prev_pixel_fract[0] * one_minus_prev_pixel_fract[1],
                                            prev_pixel_fract[0]           * one_minus_prev_pixel_fract[1],
                                            one_minus_prev_pixel_fract[0] * prev_pixel_fract[1],
                                            prev_pixel_fract[0]           * prev_pixel_fract[1]])
                        total_weight = 0.0

                        # Bilinear sampling
                        for j in range(4):
                            sample_location = prev_frame_pixel + offsets[j]
                            if sample_location[0] >= 0 and sample_location[0] < W and sample_location[1] >= 0 and sample_location[1] < H:
                                prev_world_position = world_positions[i-1][sample_location[1], sample_location[0]]
                                position_difference = (world_position[:3] - prev_world_position)
                                position_distance_squared = np.dot(position_difference, position_difference)
                                if position_distance_squared < POSITION_LIMIT_SQUARED[scene_name]:
                                    prev_normal = normals[i-1][sample_location[1], sample_location[0]]
                                    normal_difference = prev_normal - normal
                                    normal_distance_squared = np.dot(normal_difference, normal_difference)
                                    if normal_distance_squared < NORMAL_LIMIT_SQUARED[scene_name]:
                                        sample_spp += weights[j] * previous_spp[sample_location[1], sample_location[0]]
                                        previous_color += weights[j] * colors_post_accumulated[i-1][sample_location[1], sample_location[0]]
                                        total_weight += weights[j]

                        if total_weight > 0.0:
                            previous_color /= total_weight
                            sample_spp /= total_weight

                            blend_alpha = 1.0 / (sample_spp + 1.0)
                            blend_alpha = np.max([blend_alpha, BLEND_ALPHA])
                        
                    current_spp[h_i, w_i] = sample_spp + 1 if blend_alpha < 1.0 else 1.0
                    colors_post_accumulated[i][h_i, w_i] = blend_alpha * current_color + (1.0 - blend_alpha) * previous_color
            
            previous_spp = current_spp.copy()
            pyexr.write(os.path.join("dataset", scene_name, "acc_colors", "color"+str(i)+".exr"), colors_post_accumulated[i])

if __name__ == '__main__':
    process_data()