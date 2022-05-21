import os
import json
import numpy as np
import gym
from gym import spaces
from PIL import Image


def get_transforms(data_dir):
    assert os.path.exists(data_dir), "Data_dir does not exist"
    transforms_path = os.path.join(data_dir, "transforms.json")
    with open(transforms_path, "r") as f:
        transformations = json.load(f)
    return transformations


class OfflineActiveVisionEnv(gym.Env):
    def __init__(self, data_dir: str):
        """
        :params:
            data_dir: assumes data folder structure
            data_dir
                - transforms.json
                - images/
                    rgb0.png
                    depth0.png
                    rgb1.png
                    depth1.png
                    ...
        """
        self.data_dir = data_dir
        self.transforms = get_transforms(data_dir)
        self.frames = self.transforms["frames"]
        self.num_views = len(self.frames)
        self.action_space = spaces.Discrete(self.num_views)

        dummy_img = self._get_obs(self.frames[0])["img"][..., :3]
        dummy_cam2world_matrix = self._get_obs(self.frames[0])["cam2world_matrix"]

        self.observation_space = spaces.Dict(
            {
                "img": spaces.Box(
                    low=0, high=1, shape=dummy_img.shape, dtype=dummy_img.dtype
                ),
                "cam2world_matrix": spaces.Box(
                    low=-20,
                    high=20,
                    shape=dummy_cam2world_matrix.shape,
                    dtype=dummy_cam2world_matrix.dtype,
                ),
            }
        )

        self.action_space = spaces.Discrete(len(self.frames))

        self._action_to_cam2world_matrix = {
            i: frame["transform_matrix"] for i, frame in enumerate(self.frames)
        }
        # dictionary going from integer to pose

    def _get_obs(self, frame):
        rgb_path = os.path.join(self.data_dir, frame["file_path"])
        cam2world_matrix = frame["transform_matrix"]
        img = np.array(Image.open(rgb_path))
        float_img = (img / 255).astype(np.float32)

        # img = (cv2.imread(rgb_path, cv2.IMREAD_COLOR) / 255).astype(np.float32) # TODO KL
        # unclear if this read screws things up test by generating image, saving then reloading
        return {"img": float_img, "cam2world_matrix": np.array(cam2world_matrix)}

    def _get_info(self):
        return {"action_to_cam2world_matrix": self._action_to_cam2world_matrix}

    def step(self, action):
        frame = self.frames[action]
        reward = 0
        info = {}
        done = False
        return self._get_obs(frame), reward, info, done

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        action = self.np_random.integers(0, self.num_views)
        frame = self.frames[action]
        obs = self._get_obs(frame)
        info = self._get_info()
        return obs, info
