# active-vision-gym
ActiveVisionGym is a set of benchmark environments for the active view planning problem in robotics.


### Usage

First, run `pip install -e .` inside the `active-vision-gym` repo.
TODO KL add folder dependencies inside setup.py

Then, to use the provided environments in a specific file:

```
import gym
import active_vision

env = gym.make("OfflineActiveVision-v0", data_dir=<path/to/data_dir>)
```

We assume `data_dir`'s folder structure is as follows:

```
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
```

We have provided sample `data_dir` in a google drive link (TODO)

To generate your own offline dataset, follow the instructions detailed below at `Generating offline data`.

### A note about the depth image data

If a pixel has an invalid depth values or if the depth value is infinity, the depth map (and distance map) value should be set to 0 by convention (TODO link to convention). 

Helpful note from (Blenderproc docs)[https://github.com/DLR-RM/BlenderProc/blob/3f40e88b72f272a1d3159849e651d690521f2aae/docs/tutorials/renderer.md#depth-distance-and-normals]: "While distance and depth images sound similar, they are not the same: In distance images, each pixel contains the actual distance from the camera position to the corresponding point in the scene. In depth images, each pixel contains the distance between the camera and the plane parallel to the camera which the corresponding point lies on."


## Generating offline data
1. Install [blenderproc](https://github.com/DLR-RM/BlenderProc)
2. Install dcargs via `blenderproc pip install git+https://github.com/brentyi/dcargs.git`
3. Install [Shapenet](http://www.shapenet.org/)
4. You can now run `scripts/blenderproc_offline_data` to generate offline data for the gym environment^^
```
blenderproc run scripts/blenderproc_offline_data.py view-planner.num-cam-positions <num cam positions> --shapenet-path <path/to/ShapenetCoreV2> --candidate-view-radius 0.8 --obj <object>
```

Example train / eval data generation script:

```
blenderproc run scripts/blenderproc_offline_data.py --shapenet-path ../ShapeNetCore.v2/ --view-planner.num-cam-positions 3 --save-data-type eval --convert-background-to-white --save-ray-info
```

`--save-ray-info`: saves rays for ray distance supervision (c.f. [DS-NeRF](https://github.com/dunbar12138/DSNeRF))

`--convert-background-to-white`: converts (infinite distance) background to RGB color `(255, 255, 255)`

> **_NOTE:_** dcargs replaces underscores with dashes on the command line, and supports nested args (e.g. `view-planner.num-cam-positions`)

The script found in `blenderproc_offline_data.py` creates a directory `<offline gym data dir>/<save data type>/<shapenet object name>`. The folder structure within this directory follows that of [nerf](https://github.com/bmild/nerf), ie.  

    transforms_train.json
    transforms_val.json
    images/
        img_0.png
        img_1.png
        ...

These instructions have been tested extensively on Ubuntu 18.04.
