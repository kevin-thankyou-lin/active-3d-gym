import os
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

import json

import numpy as np
import gym
from gym import spaces
from PIL import Image
import cv2
import imageio


def get_transforms(data_dir):
    assert os.path.exists(data_dir), f"data_dir {data_dir} does not exist"
    transforms_path = os.path.join(data_dir, "transforms_train.json")
    with open(transforms_path, "r") as f:
        transformations = json.load(f)
    return transformations


def get_object_bounds(data_dir):
    print(f"get_object_bounds is using heuristic at the moment")
    poi = np.zeros(3)
    size = 1.5
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
    assert os.path.exists(data_dir), f"data_dir {data_dir} does not exist"
    object_bounds_path = os.path.join(data_dir, "object_bounds.json")
    with open(object_bounds_path, "r") as f:
        object_bounds = json.load(f)
    return object_bounds


class OfflineActiveVisionEnv(gym.Env):
    def __init__(self, data_dir: str):
        """
        :params:
            data_dir: assumes data folder structure
            data_dir/
                - transforms.json
                - object_bounds.json # include object bound information?
                - images/
                    im_0.png
                    im_0_depth.exr
                    im_0_distance.exr
                    im_1.png
                    im_1_depth.exr
                    im_1_distance.exr
                    ...

        """
        self.data_dir = data_dir
        self.transforms = get_transforms(data_dir)
        self.object_bounds = get_object_bounds(data_dir)
        self.all_possible_frames = self.transforms["frames"]
        self.curr_episode_frames = []  # populated within reset()
        self.num_total_views = len(self.all_possible_frames)
        self.action_space = spaces.Discrete(self.num_total_views)
        dummy_obs = self._get_obs(self.all_possible_frames[0])
        dummy_img = dummy_obs["img"][..., :3]
        dummy_distance = dummy_obs["distance"]
        dummy_cam2world_matrix = dummy_obs["cam2world_matrix"]

        self.observation_space = spaces.Dict(
            {
                "img": spaces.Box(
                    low=0, high=1, shape=dummy_img.shape, dtype=dummy_img.dtype
                ),
                "distance": spaces.Box(
                    low=0,
                    high=1000,
                    shape=dummy_distance.shape if dummy_distance is not None else (0,),
                    dtype=dummy_distance.dtype if dummy_distance is not None else np.float32,
                ),
                "cam2world_matrix": spaces.Box(
                    low=-1,
                    high=1,
                    shape=dummy_cam2world_matrix.shape,
                    dtype=dummy_cam2world_matrix.dtype,
                ),
            }
        )

        self.action_space = spaces.Discrete(len(self.all_possible_frames))

        self._action_to_cam2world_matrix = {
            i: frame["transform_matrix"]
            for i, frame in enumerate(self.all_possible_frames)
        }
        self.curr_episode_action_to_cam2world_matrix = {}  # populated within reset()

        self.reset_counter = 0
        # dictionary going from integer to pose

    def _get_obs(self, frame):
        rgb_path = os.path.join(self.data_dir, frame["file_path"])
        if "." not in rgb_path.split("/")[-1]:
            rgb_path += ".png"
        cam2world_matrix = frame["transform_matrix"]
        img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
        float_img = (img / 255).astype(np.float32)
        if "distance_path" in frame:
            distance_path = os.path.join(self.data_dir, frame["distance_path"])
            distance = imageio.imread(distance_path)
        else:
            distance = None

        return {
            "img": float_img,
            "distance": distance,
            "cam2world_matrix": np.array(cam2world_matrix),
        }

    def _get_info(self):
        transform = self.transforms
        if "camera_angle_x" in transform or "camera_angle_y" in transform:
            self.H = self.observation_space["img"].shape[0]
            self.W = self.observation_space["img"].shape[1]
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = (
                self.W / (2 * np.tan(transform["camera_angle_x"] / 2))
                if "camera_angle_x" in transform
                else None
            )
            fl_y = (
                self.H / (2 * np.tan(transform["camera_angle_y"] / 2))
                if "camera_angle_y" in transform
                else None
            )
            if fl_x is None:
                fl_x = fl_y
            if fl_y is None:
                fl_y = fl_x

            cx = (transform["cx"] / downscale) if "cx" in transform else (self.H / 2)
            cy = (transform["cy"] / downscale) if "cy" in transform else (self.W / 2)

            self.transforms["fl_x"] = fl_x
            self.transforms["fl_y"] = fl_y
            self.transforms["c_x"] = cx
            self.transforms["c_y"] = cy

        camera_info_dct = {
            "fl_x": self.transforms["fl_x"],
            "fl_y": self.transforms["fl_y"],
            "c_x": self.transforms["c_x"],
            "c_y": self.transforms["c_y"],
            "height": self.observation_space["img"].shape[0],
            "width": self.observation_space["img"].shape[1],
        }
        return {
            "action_to_cam2world_matrix": self.curr_episode_action_to_cam2world_matrix,
            "camera_info": camera_info_dct,
            "object_bounds": self.object_bounds,  # tODO KL update this
        }

    def _update_action_space(self, actions: List[int]):
        """Update action_space, curr_episode_frames, curr_episode_action_to_cam2world_matrix based on the initial 'actions' taken
        - only used after reset()

        :params:
            actions: list of integers denoting used actions, each integer is the index of the frame in self.all_possible_frames
        """
        self.curr_episode_frames = [
            frame
            for idx, frame in enumerate(self.all_possible_frames)
            if idx not in actions
        ]
        self.action_space = spaces.Discrete(len(self.curr_episode_frames))
        self.curr_episode_action_to_cam2world_matrix = {
            i: frame["transform_matrix"]
            for i, frame in enumerate(self.curr_episode_frames)
        }

    def step(self, action: int):
        frame = self.curr_episode_frames[action]
        reward = 0
        info = {"duration": 0}
        done = False
        return self._get_obs(frame), reward, done, info

    def reset(self, seed=None, return_info=False, options: Optional[dict] = {}):
        """
        :params:
            options: supports key of "num_init_views" and value of int
                specifies the number of initial views to be returned by the environment
                on reset. If not specified, the default is to return one random view.
        :returns:
            info, a dict with keys
                - "init_views"
                - "action_to_cam2world_matrix"
                - "camera_info"
                - "object_bounds"

            obs: dict containing the last observation in the initial sequence of views
        """
        super().reset(seed=seed)
        num_init_views = options.get("num_init_views", 1)
        actions = self.np_random.integers(0, self.num_total_views, size=num_init_views)
        print(
            f"[INFO] init_view actions: {actions}: these actions will be removed from the action space"
        )
        frames = np.array(self.all_possible_frames)[np.array(actions)]
        self._update_action_space(actions)

        obs_lst = [self._get_obs(frame) for frame in frames]
        info = self._get_info()
        info["init_views"] = obs_lst
        if return_info:
            return obs_lst[-1], info
        else:
            return obs_lst[-1]
