# env/__init__.py

from gymnasium.envs.registration import register, registry

# register only once
if "CoMEC-v0" not in registry:
    register(
        id="CoMEC-v0",
        entry_point="env.env:CoMECGymEnv",
        max_episode_steps=200,
    )
