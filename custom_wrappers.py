from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import (
    EpisodeWrapper,
    VmapWrapper,
    DomainRandomizationVmapWrapper,
)
import jax
from jax import numpy as jp
from mujoco import mjx


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[System], Tuple[System, System]]] = None,
) -> Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    env = EpisodeWrapper(env, episode_length, action_repeat)
    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = AutoResetWrapperTracking(env)
    return env


class AutoResetWrapperTracking(Wrapper):
    """Automatically resets RodentMultiClipTracking envs that are done.

    Each reset selects a new random clip_idx to ensure varied initial conditions.
    """

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment and initializes the 'first_' info."""
        state = self.env.reset(rng)
        # Save rng
        state.info["reset_rng"] = rng
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        # # Split the RNG from the current state
        # split_keys = jax.vmap(jax.random.split, in_axes=(0, None))(
        #     state.info["reset_rng"], 2
        # )
        # rng, reset_rng = split_keys[:, 0], split_keys[:, 1]
        info = state.info

        # Define the function to perform a reset
        def reset_fn(rng):
            # Reset the environment, which selects a new random clip_idx
            rng, start_rng, clip_rng, rng1, rng2 = jax.random.split(rng, 5)

            start_frame = jax.random.randint(start_rng, (), 0, 44)
            clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)

            new_info = {
                "clip_idx": clip_idx,
                "cur_frame": start_frame,
                "steps_taken_cur_frame": 0,
                "summed_pos_distance": 0.0,
                "quat_distance": 0.0,
                "joint_distance": 0.0,
            }

            # Get reference clip and select the start frame
            reference_frame = jax.tree_map(
                lambda x: x[info["cur_frame"]], self._get_reference_clip(info)
            )

            low, hi = -self._reset_noise_scale, self._reset_noise_scale

            # Add pos
            qpos_with_pos = (
                jp.array(self.sys.qpos0).at[:3].set(reference_frame.position)
            )

            # Add quat
            new_qpos = qpos_with_pos.at[3:7].set(reference_frame.quaternion)

            # Add noise
            qpos = new_qpos + jax.random.uniform(
                rng1, (self.sys.nq,), minval=low, maxval=hi
            )
            qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

            data = mjx.forward(state.pipeline_state.replace(qpos, qvel))

            obs = self._get_obs(data, info)

            state.replace(pipeline_state=data, obs=obs, info=info)
            # Update info with specific new info content while retaining the rest
            info["clip_idx"] = new_info["clip_idx"]
            info["cur_frame"] = new_info["cur_frame"]
            info["steps_taken_cur_frame"] = new_info["steps_taken_cur_frame"]
            info["reset_rng"] = rng

            return state.replace(pipeline_state=data, obs=obs, info=info)

        # Define the function to retain the next_state without resetting
        def no_reset_fn(rng):
            # Update the RNG in the 'info' for future splits
            return state

        done = state.done
        rng = info["reset_rng"]
        print(done.shape, rng.shape)
        # Use JAX's conditional to decide whether to reset or not
        return jax.vmap(
            lambda done, rng: jax.lax.cond(done, reset_fn, no_reset_fn, operand=rng),
            in_axes=(0, 0),
        )(
            done,
            rng,
        )


class RenderRolloutWrapperTracking(Wrapper):
    """Always resets to 0"""

    def reset(self, rng: jax.Array) -> State:
        _, clip_rng, rng = jax.random.split(rng, 4)

        clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)
        info = {
            "clip_idx": clip_idx,
            "cur_frame": 0,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
        }

        return self.reset_from_clip(rng, info)


# Single clip
# class AutoResetWrapperTracking(Wrapper):
#     """Automatically resets Brax envs that are done."""

#     def reset(self, rng: jax.Array) -> State:
#         state = self.env.reset(rng)
#         state.info["first_pipeline_state"] = state.pipeline_state
#         state.info["first_obs"] = state.obs
#         state.info["first_cur_frame"] = state.info["cur_frame"]
#         state.info["first_steps_taken_cur_frame"] = state.info["steps_taken_cur_frame"]
#         return state

#     def step(self, state: State, action: jax.Array) -> State:
#         if "steps" in state.info:
#             steps = state.info["steps"]
#             steps = jp.where(state.done, jp.zeros_like(steps), steps)
#             state.info.update(steps=steps)
#         state = state.replace(done=jp.zeros_like(state.done))
#         state = self.env.step(state, action)

#         def where_done(x, y):
#             done = state.done
#             if done.shape:
#                 done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
#             return jp.where(done, x, y)

#         pipeline_state = jax.tree.map(
#             where_done, state.info["first_pipeline_state"], state.pipeline_state
#         )
#         obs = where_done(state.info["first_obs"], state.obs)
#         state.info["cur_frame"] = where_done(
#             state.info["first_cur_frame"],
#             state.info["cur_frame"],
#         )
#         state.info["steps_taken_cur_frame"] = where_done(
#             state.info["first_steps_taken_cur_frame"],
#             state.info["steps_taken_cur_frame"],
#         )
#         return state.replace(pipeline_state=pipeline_state, obs=obs)


# Single clip
# class RenderRolloutWrapperTracking(Wrapper):
#     """Always resets to 0"""

#     def reset(self, rng: jax.Array) -> State:
#         info = {
#             "cur_frame": 0,
#             "steps_taken_cur_frame": 0,
#             "summed_pos_distance": 0.0,
#             "quat_distance": 0.0,
#             "joint_distance": 0.0,
#         }

#         return self.reset_from_clip(rng, info)
