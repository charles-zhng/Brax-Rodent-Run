import functools
import jax
from typing import Dict
import wandb
import imageio
import mujoco
from brax import envs
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale

from brax.io import model
import numpy as np
from rodent import RodentJoystick
import pickle
import warnings
from preprocessing.mjx_preprocess import process_clip_to_train
from jax import numpy as jp
import orbax.checkpoint as ocp

import custom_ppo as ppo
import custom_wrappers
from custom_losses import PPONetworkParams
import custom_ppo_networks
import network_masks as masks
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from absl import app
from absl import flags

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["MUJOCO_GL"] = "egl"

FLAGS = flags.FLAGS

try:
    n_devices = jax.device_count(backend="gpu")
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
    )
    print(f"Using {n_devices} GPUs")
except:
    n_devices = 1
    print("Not using GPUs")


config = {
    "env_name": "joystick",
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 256 * n_devices,
    "num_timesteps": 1_000_000_000,
    "eval_every": 10_000_000,
    "episode_length": 500,
    "batch_size": 256 * n_devices,
    "num_minibatches": 4 * n_devices,
    "num_updates_per_batch": 4,
    "learning_rate": 3e-4,
    "kl_loss": False,
    "clipping_epsilon": 0.2,
    "torque_actuators": True,
    "run_platform": "Harvard",
    "solver": "cg",
    "iterations": 8,
    "ls_iterations": 8,
}

envs.register_environment("joystick", RodentJoystick)

clip_id = -1

# instantiate the environment
env = envs.get_environment(
    config["env_name"],
    torque_actuators=config["torque_actuators"],
    solver=config["solver"],
    iterations=config["iterations"],
    ls_iterations=config["ls_iterations"],
)

# Episode length is equal to (clip length - random init range - traj length) * steps per cur frame
# Will work on not hardcoding these values later

# Define mask for freezing weights
# mask = {"params": {"encoder": "encoder", "decoder": "decoder", "bottleneck": "encoder"}}
# value = {"params": "encoder"}
# freeze_mask = PPONetworkParams(mask, value)


train_fn = functools.partial(
    ppo.train,
    num_timesteps=config["num_timesteps"],
    num_evals=int(config["num_timesteps"] / config["eval_every"]),
    num_resets_per_eval=1,
    reward_scaling=1,
    episode_length=500,
    normalize_observations=True,
    action_repeat=1,
    clipping_epsilon=config["clipping_epsilon"],
    unroll_length=20,
    num_minibatches=config["num_minibatches"],
    num_updates_per_batch=config["num_updates_per_batch"],
    discounting=0.95,
    learning_rate=config["learning_rate"],
    kl_loss=config["kl_loss"],
    entropy_cost=1e-2,
    num_envs=config["num_envs"],
    batch_size=config["batch_size"],
    seed=0,
    network_factory=functools.partial(
        custom_ppo_networks.make_encoderdecoder_ppo_networks,
        intention_latent_size=60,
        encoder_hidden_layer_sizes=(512, 512),
        decoder_hidden_layer_sizes=(512, 512),
        value_hidden_layer_sizes=(512, 512),
    ),
    checkpoint_network_factory=functools.partial(
        custom_ppo_networks.make_intention_ppo_networks,
        intention_latent_size=60,
        encoder_hidden_layer_sizes=(512, 512),
        decoder_hidden_layer_sizes=(512, 512),
        value_hidden_layer_sizes=(512, 512),
    ),
    freeze_mask_fn=masks.create_decoder_mask,
    checkpoint_path=Path(
        "/Users/charleszhang/GitHub/Brax-Rodent-Run/943aa149-d90f-46c8-b57c-1c2a0a93031d-latest"
    ),
    continue_training=False,
)

import uuid

# Generates a completely random UUID (version 4)
run_id = uuid.uuid4()
checkpoint_dir = os.path.abspath(f"./model_checkpoints/{run_id}")

options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
ckpt_mgr = ocp.CheckpointManager(
    checkpoint_dir,
    item_names=("normalizer_params", "params", "env_steps"),
    options=options,
)

run = wandb.init(project="joystick_rat", config=config, notes=f"")


wandb.run.name = (
    f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{run_id}"
)


def wandb_progress(num_steps, metrics):
    metrics["num_steps"] = num_steps
    wandb.log(metrics, commit=False)


# Wrap the env in the brax autoreset and episode wrappers
rollout_env = env
# define the jit reset/step functions
jit_reset = jax.jit(rollout_env.reset)
jit_step = jax.jit(rollout_env.step)


def policy_params_fn(
    num_steps, make_policy, params, rollout_key, checkpoint_dir=checkpoint_dir
):
    (processor_params, network_params, env_steps) = params
    print(network_params.policy["params"]["decoder"]["hidden_0"]["kernel"])
    jit_inference_fn = jax.jit(
        make_policy((processor_params, network_params.policy), deterministic=True)
    )
    rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)

    state = jit_reset(reset_rng)

    rollout = [state]
    for i in range(int(500)):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        ctrl, extras = jit_inference_fn(obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)

    # Render the walker with the reference expert demonstration trajectory
    os.environ["MUJOCO_GL"] = "osmesa"

    # Render rollout
    video_path = f"{checkpoint_dir}/{num_steps}.mp4"

    imageio.mimwrite(
        video_path,
        rollout_env.render(rollout, camera=1, height=500, width=500),
        fps=int((1.0 / env.dt)),
    )

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})


make_inference_fn, params, _ = train_fn(
    environment=env,
    progress_fn=wandb_progress,
    policy_params_fn=policy_params_fn,
    checkpoint_manager=ckpt_mgr,
)

final_save_path = f"{checkpoint_dir}/brax_ppo_rodent_run_finished"
model.save_params(final_save_path, params)
print(f"Run finished. Model saved to {final_save_path}")
