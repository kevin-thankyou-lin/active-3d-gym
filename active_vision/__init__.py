from gym.envs.registration import register

register(
    id="OfflineActiveVision-v0",
    entry_point="active_vision.envs:OfflineActiveVisionEnv",
)
