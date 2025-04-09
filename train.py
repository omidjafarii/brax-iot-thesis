import jax
from brax import envs
from brax.training import ppo

env = envs.get_environment('ant')

inference_fn, params, metrics = ppo.train(
    environment=env,
    num_timesteps=500_000,
    episode_length=1000,
    num_envs=64,
    seed=0,
    log_training_metrics=True,
    training_metrics_steps=10000
)

print("Training complete.")
