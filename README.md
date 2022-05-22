# active-vision-gym
ActiveVisionGym is a set of benchmark environments for the active view planning problem in robotics.

## Generating offline data
1. Install [blenderproc](https://github.com/DLR-RM/BlenderProc)
2. Install dcargs via `blenderproc pip install git+https://github.com/brentyi/dcargs.git`
3. Install [Shapenet](http://www.shapenet.org/)
4. You can now run `scripts/blenderproc_offline_data` to generate offline data for the gym environment^^
```
blenderproc run scripts/blenderproc_offline_data.py view-planner.num-cam-positions <num cam positions> --shapenet-path <path/to/ShapenetCoreV2> --candidate-view-radius 0.8 --obj <object>
```
> **_NOTE:_** dcargs replaces underscores with dashes on the command line, and supports nested args (e.g. `view-planner.num-cam-positions`)

The script found in `blenderproc_offline_data.py` creates a directory `<offline gym data dir>/<save data type>/<shapenet object name>`. The folder structure within this directory follows that of [nerf](https://github.com/bmild/nerf), ie.  

    transforms_train.json
    transforms_val.json
    images/
        img_0.png
        img_1.png
        ...

These instructions have been tested extensively on Ubuntu 18.04.
