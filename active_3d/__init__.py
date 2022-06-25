from gym.envs.registration import register

register(
    id="OfflineActive3D-v0",
    entry_point="active_3d.envs:OfflineActive3DEnv",
)
