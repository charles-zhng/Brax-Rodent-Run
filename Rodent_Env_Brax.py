import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm

import mujoco
from mujoco import mjx

import numpy as np

import os

_XML_PATH = "./models/rodent_new.xml"


class Rodent(PipelineEnv):

    def __init__(
        self,
        track_pos: jp.ndarray,
        track_quat: jp.ndarray,
        ref_len: int = 5,
        forward_reward_weight=10,
        ctrl_cost_weight=0.01,
        pos_reward_weight=100.0,
        quat_reward_weight=3.0,
        healthy_reward=0.25,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.04, 0.5),
        reset_noise_scale=1e-3,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        **kwargs,
    ):
        # Load the rodent model via dm_control
        # dm_rodent = rodent.Rodent()
        # physics = mjcf_dm.Physics.from_mjcf_model(dm_rodent.mjcf_model)
        # mj_model = physics.model.ptr
        root = mjcf_dm.from_path(_XML_PATH)
        rescale.rescale_subtree(
            root,
            0.9,
            0.9,
        )
        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
        os.environ["MUJOCO_GL"] = "osmesa"
        mj_model = mujoco.MjModel.from_xml_path(_XML_PATH)
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations

        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        physics_steps_per_control_step = (
            10  # 10 times 0.002 = 0.02 => fps of tracking data
        )

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._torso_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), "torso"
        )

        self._track_pos = track_pos
        self._track_quat = track_quat
        self._ref_len = ref_len
        self._pos_reward_weight = pos_reward_weight
        self._quat_reward_weight = quat_reward_weight
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng_pos = jax.random.split(rng, 4)

        start_frame = jax.random.randint(rng, (), 0, 44)

        info = {
            "cur_frame": start_frame,
        }

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Add pos (without z height)
        qpos_with_pos = (
            jp.array(self.sys.qpos0).at[:2].set(self._track_pos[start_frame][:2])
        )

        # Add quat
        new_qpos = qpos_with_pos.at[3:7].set(self._track_quat[start_frame])

        # Add noise
        qpos = new_qpos + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, start_frame)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "pos_reward": zero,
            "quat_reward": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
        }
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        info = state.info.copy()
        info["cur_frame"] += 1
        pos_reward = self._pos_reward_weight * jp.exp(
            -400 * jp.sum(data.qpos[:3] - self._track_pos[info["cur_frame"]]) ** 2
        )
        quat_reward = self._quat_reward_weight * jp.exp(
            -4
            * jp.sum(
                self._bounded_quat_dist(
                    data.qpos[3:7], self._track_quat[info["cur_frame"]]
                )
                ** 2
            )
        )

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._torso_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._torso_idx][2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, info["cur_frame"])
        reward = pos_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            pos_reward=pos_reward,
            quat_reward=quat_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_obs(self, data: mjx.Data, cur_frame: int) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""
        track_pos_local = jax.vmap(self.emil_to_local, in_axes=(None, 0))(
            self._track_pos[cur_frame + 1 : cur_frame + 1 + self._ref_len]
            - data.qpos[:3]
        ).flatten()
        # get relative tracking position in local frame
        # track_pos_local = self.emil_to_local(
        #     data,
        # ).flatten()

        quat_dist = jax.vmap(self._bounded_quat_dist, in_axes=(None, 0))(
            data.qpos[3:7],
            self._track_quat[cur_frame + 1 : cur_frame + 1 + self._ref_len],
        ).flatten()

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                data.qpos,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
                track_pos_local,
                quat_dist,
            ]
        )

    def emil_to_local(self, data, vec_in_world_frame):
        xmat = jp.reshape(data.xmat[0], (3, 3))
        return xmat @ vec_in_world_frame

    def to_local(self, data, vec_in_world_frame):
        """Linearly transforms a world-frame vector into entity's local frame.

        Note that this function does not perform an affine transformation of the
        vector. In other words, the input vector is assumed to be specified with
        respect to the same origin as this entity's local frame. This function
        can also be applied to matrices whose innermost dimensions are either 2 or
        3. In this case, a matrix with the same leading dimensions is returned
        where the innermost vectors are replaced by their values computed in the
        local frame.

        Returns the resulting vector, converting to ego-centric frame
        """
        # TODO: confirm index
        xmat = jp.reshape(data.xmat[1], (3, 3))
        # The ordering of the np.dot is such that the transformation holds for any
        # matrix whose final dimensions are (2,) or (3,).

        # Each element in xmat is a 3x3 matrix that describes the rotation of a body relative to the global coordinate frame, so
        # use rotation matrix to dot the vectors in the world frame, transform basis
        if vec_in_world_frame.shape[-1] == 2:
            return jp.dot(vec_in_world_frame, xmat[:2, :2])
        elif vec_in_world_frame.shape[-1] == 3:
            return jp.dot(vec_in_world_frame, xmat)
        else:
            raise ValueError(
                "`vec_in_world_frame` should have shape with final "
                "dimension 2 or 3: got {}".format(vec_in_world_frame.shape)
            )

    def _bounded_quat_dist(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Computes a quaternion distance limiting the difference to a max of pi/2.

        This function supports an arbitrary number of batch dimensions, B.

        Args:
          source: a quaternion, shape (B, 4).
          target: another quaternion, shape (B, 4).

        Returns:
          Quaternion distance, shape (B, 1).
        """
        source /= jp.linalg.norm(source, axis=-1, keepdims=True)
        target /= jp.linalg.norm(target, axis=-1, keepdims=True)
        # "Distance" in interval [-1, 1].
        dist = 2 * jp.einsum("...i,...i", source, target) ** 2 - 1
        # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
        dist = jp.minimum(1.0, dist)
        # Divide by 2 and add an axis to ensure consistency with expected return
        # shape and magnitude.
        return 0.5 * jp.arccos(dist)[..., np.newaxis]
