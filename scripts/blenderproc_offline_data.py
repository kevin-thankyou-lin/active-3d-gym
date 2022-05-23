import blenderproc as bproc
import dataclasses
import dcargs
import json
import numpy as np
import os
from PIL import Image
import shutil
from typing import Literal
import imageio

from blenderproc.python.camera.CameraUtility import get_intrinsics_as_K_matrix

# Object to shapenet id mappings
obj_to_shapenet_ids = {
    "Airplane": {
        "used_synset_id": "02691156",
        "used_source_id": "8f4e31ee9912f54e77fd7318510b8627",
    },
    "Bag": {
        "used_synset_id": "02773838",
        "used_source_id": "4e4fcfffec161ecaed13f430b2941481",
    },
    "Car": {
        "used_synset_id": "02958343",
        "used_source_id": "1ae9732840a315afab2c2809513f396e",
    },
    "Guitar": {
        "used_synset_id": "03467517",
        "used_source_id": "4ff4b6305e3d5e2ad57700c05b1862d8",
    },
    "Lamp": {
        "used_synset_id": "03636649",
        "used_source_id": "8ef9c1ffaa6f7292bd73284bc3c3cbac",
    },
    "Cap": {
        "used_synset_id": "02954340",
        "used_source_id": "188702a3bb44f40c284432ce2f42f498",
    },
    "Motorbike": {
        "used_synset_id": "03790512",
        "used_source_id": "6bf1f988493dc23752ee90e577613070",
    },
    "Skateboard": {
        "used_synset_id": "04225987",
        "used_source_id": "619b49bae27807fb1082f2ea630bf69e",  #
    },
    "Camera": {
        "used_synset_id": "02942699",
        "used_source_id": "97690c4db20227d248e23e2c398d8046",
    },
}


@dataclasses.dataclass
class ViewPlanner:
    candidate_view_radius: float = (
        1.3  # heuristic: account for workspace size and arm kinematics
    )
    candidate_view_type: Literal["hemisphere", "circle", "sphere"] = "sphere"
    num_cam_positions: int = 100


@dataclasses.dataclass
class Args:
    #############################
    # Blenderproc rendering args #
    #############################
    view_planner: ViewPlanner = None
    shapenet_path: str = ""  # Path to the downloaded shape net core v2 dataset, get it from http://www.shapenet.org/
    offline_gym_data_dir: str = "../active-nerf/offline_gym_data"  # TODO KL update Path to where the final files will be saved
    save_data_type: Literal["train", "eval"] = "eval"
    save_ray_info: bool = False  # Ray info (w.r.t a frame) consists of ray_coord, ray_length and ray weight
    convert_background_to_white: bool = False
    obj: Literal[
        "Airplane",
        "Bag",
        "Car",
        "Guitar",
        "Lamp",
        "Cap",
        "Motorbike",
        "Skateboard",
        "Camera",
    ] = "Camera"


##################### active nerf utils ###############################
def fibonacci_sphere(samples: int) -> np.ndarray:
    """Fibonacci sphere algorithm for uniform sampling on a sphere. Returns array with
    shape (samples, 3).

    Adapted from:
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere"""
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians.
    i = np.arange(samples)
    y = 1.0 - (i / float(samples - 1.0)) * 2.0  # y goes from 1 to -1.
    radius = np.sqrt(1 - y * y)  # Radius at y.
    theta = phi * i  # Golden angle increment.
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.array([x, y, z]).T


def get_candidate_camera_positions_circle(
    radius: float,
    cam_z: float,
    num_cam_positions: int,
    center_x: float = 0,
    center_y: float = 0,
) -> dict:
    """Returns a dictionary with keys as camera angles and values as camera positions."""
    cam_positions = {}
    single_angle = 360 / num_cam_positions
    for i in range(num_cam_positions):
        x = radius * np.cos(2 * np.pi / num_cam_positions * i) + center_x
        y = radius * np.sin(2 * np.pi / num_cam_positions * i) + center_y
        cam_positions[int(single_angle * i)] = np.array((x, y, cam_z))
    return cam_positions


def get_candidate_views(
    object_bounds: dict,
    num_cam_positions: int,
    candidate_view_radius: float = 0.8,
    candidate_view_type: str = "sphere",
):
    """Returns a list of openGL camera poses that point towards the object.
    Radius should come from object scene bounds
    """
    poi = object_bounds["poi"]
    size = object_bounds["size"]
    assert candidate_view_radius > size * np.sqrt(
        2
    ), "Sphere radius must be larger than object size"

    if candidate_view_type == "hemisphere":
        sphere_radius = candidate_view_radius
        fib_xyzs = fibonacci_sphere(2 * num_cam_positions)
        # heuristic: filter out values whose z value is below e.g. 0.25 to look from above
        filtered_xyzs = fib_xyzs[np.where(fib_xyzs[:, -1] > 0.4)]
        rescaled_xyzs = sphere_radius * filtered_xyzs
    elif candidate_view_type == "circle":
        circle_radius = candidate_view_radius * (
            object_bounds["xmax"] - object_bounds["xmin"]
        )  # heuristic: extra distance from object
        cam_z = poi[2]
        rescaled_xyzs = get_candidate_camera_positions_circle(
            circle_radius, cam_z, num_cam_positions
        )
        rescaled_xyzs = np.array(list(rescaled_xyzs.values()))
    elif candidate_view_type == "sphere":
        sphere_radius = candidate_view_radius
        fib_xyzs = fibonacci_sphere(num_cam_positions)
        rescaled_xyzs = sphere_radius * fib_xyzs

    world_xyzs = rescaled_xyzs + poi
    # calculate the camera poses
    openGL_cam_positions = []
    for loc in world_xyzs:
        rotation_mat = bproc.camera.rotation_from_forward_vec(poi - loc)
        openGL_cam_positions.append(
            bproc.math.build_transformation_mat(loc, rotation_mat)
        )
    return openGL_cam_positions


def get_shapenet_obj_bounds(shapenet_obj):
    blender_obj = shapenet_obj.blender_obj
    poi = blender_obj.location[:]
    bound_vals = blender_obj.bound_box[:]
    size = np.array([np.array(i) for i in list(bound_vals)]).max()

    return {
        "poi": poi,
        "size": size,
        "xmin": poi[0] - 2 * size,
        "xmax": poi[0] + 2 * size,
        "ymin": poi[1] - 2 * size,
        "ymax": poi[1] + 2 * size,
        "zmin": poi[2] - 2 * size,
        "zmax": poi[2] + 2 * size,
    }  # 2 * is rough heuristic


def get_transformations_template(
    fl_x=None, fl_y=None, c_x=None, c_y=None, aabb_scale=None, frames=[]
):
    transformations = {
        "fl_x": fl_x,
        "fl_y": fl_y,
        "c_x": c_x,
        "c_y": c_y,
        "aabb_scale": aabb_scale,
        "frames": frames,
    }
    return transformations


def openGL_to_openCV(cam2world_matrix):
    tmat = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-0.0, -1.0, -0.0, -0.0],
            [-0.0, -0.0, -1.0, -0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return np.matmul(cam2world_matrix, tmat)


############################# end of active nerf utils ################################


def main(args):
    offline_gym_obj_dir = os.path.join(
        args.offline_gym_data_dir, args.save_data_type, args.obj
    )
    if os.path.isdir(offline_gym_obj_dir):
        shutil.rmtree(offline_gym_obj_dir)
    os.makedirs(offline_gym_obj_dir)

    bproc.init()

    intrinsics = get_intrinsics_as_K_matrix()
    transformations_template = get_transformations_template(
        fl_x=intrinsics[0][0],
        fl_y=intrinsics[1][1],
        c_x=intrinsics[0][2],
        c_y=intrinsics[1][2],
    )

    # Load the ShapeNet object into the scene
    shapenet_obj_ids = obj_to_shapenet_ids[args.obj]
    shapenet_obj = bproc.loader.load_shapenet(
        args.shapenet_path,
        used_synset_id=shapenet_obj_ids["used_synset_id"],
        used_source_id=shapenet_obj_ids["used_source_id"],
        move_object_origin=False,
    )

    # Define lights and set location and energy level
    light1 = bproc.types.Light()
    light1.set_type("POINT")
    light1.set_location([0, 0, 5])
    light1.set_energy(200)

    light2 = bproc.types.Light()
    light2.set_type("POINT")
    light2.set_location([0, 5, 0])
    light2.set_energy(200)

    light3 = bproc.types.Light()
    light3.set_type("POINT")
    light3.set_location([0, -5, 0])
    light3.set_energy(200)

    light4 = bproc.types.Light()
    light4.set_type("POINT")
    light4.set_location([5, 0, 0])
    light4.set_energy(200)

    light5 = bproc.types.Light()
    light5.set_type("POINT")
    light5.set_location([-5, 0, 0])
    light5.set_energy(200)

    # Calculate the camera poses
    object_bounds = get_shapenet_obj_bounds(shapenet_obj)
    openGL_cam_positions = get_candidate_views(
        object_bounds,
        args.view_planner.num_cam_positions,
        args.view_planner.candidate_view_radius,
        args.view_planner.candidate_view_type,
    )

    # Add camera poses to scene
    for i in range(len(openGL_cam_positions)):
        openGL_transform_matrix = openGL_cam_positions[i]
        bproc.camera.add_camera_pose(openGL_transform_matrix)

    # Render the whole pipeline
    # bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    data = bproc.renderer.render()

    image_dir = f"{offline_gym_obj_dir}/images"
    os.makedirs(image_dir)  # making directory for each image

    for i in range(len(openGL_cam_positions)):
        rgb = np.array(data["colors"][i])  # [..., :3] # remove alpha channel
        depth = np.array(data["depth"][i])
        distance = bproc.postprocessing.depth2dist(depth).astype(np.float32)

        if args.convert_background_to_white:
            rgb[depth > 1e8] = 255

        depth[depth > 1e8] = 0
        distance[distance > 1e8] = 0

        rgb_png = Image.fromarray(rgb)
        rgb_path = f"images/im_{i}.png"
        rgb_png.save(os.path.join(offline_gym_obj_dir, rgb_path))

        depth_exr = depth
        depth_path = f"images/im_{i}_depth.exr"
        imageio.imwrite(os.path.join(offline_gym_obj_dir, depth_path), depth_exr)

        distance_exr = distance
        distance_path = f"images/im_{i}_distance.exr"
        imageio.imwrite(os.path.join(offline_gym_obj_dir, distance_path), distance_exr)

        transform_dct = {
            "file_path": rgb_path,
            "depth_path": depth_path,
            "distance_path": distance_path,
            "transform_matrix": openGL_cam_positions[i].tolist(),
        }

        if args.save_ray_info:
            ray_lengths_path = f"images/ray_lengths_{i}.npy"
            ray_coords_path = f"images/ray_coords_{i}.npy"
            ray_weights_path = f"images/ray_weights_{i}.npy"

            ray_lengths = distance.copy()
            ray_coords = np.meshgrid(
                np.arange(distance.shape[0]), np.arange(distance.shape[1])
            )
            ray_weights = np.ones_like(distance)
            ray_weights[distance == 0] = 0  # do not supervise inf distance rays
            ray_info_dct = {
                "ray_lengths": ray_lengths_path,
                "ray_coords": ray_coords_path,
                "ray_weights": ray_weights_path,
            }

            np.save(os.path.join(offline_gym_obj_dir, ray_lengths_path), ray_lengths)
            np.save(os.path.join(offline_gym_obj_dir, ray_coords_path), ray_coords)
            np.save(os.path.join(offline_gym_obj_dir, ray_weights_path), ray_weights)

            transform_dct.update(ray_info_dct)

        transformations_template["frames"].append(transform_dct)

    for data_type in ["train", "val"]:
        with open(f"{offline_gym_obj_dir}/transforms_{data_type}.json", "w") as f:
            json.dump(transformations_template, f)


if __name__ == "__main__":
    args = dcargs.parse(Args)
    main(args)
