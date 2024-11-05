import jax
from jax import numpy as jp

from brax.base import Base, Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from brax import base
from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm
from ml_collections import config_dict

from jax.numpy import inf, ndarray
import mujoco
from mujoco import mjx

import numpy as np

from preprocessing.mjx_preprocess import ReferenceClip

from typing import Any

_XML_PATH = "./models/rodent.xml"
_MOCAP_HZ = 50
_JOINT_NAMES = [
    "vertebra_1_extend",
    "hip_L_supinate",
    "hip_L_abduct",
    "hip_L_extend",
    "knee_L",
    "ankle_L",
    "toe_L",
    "hip_R_supinate",
    "hip_R_abduct",
    "hip_R_extend",
    "knee_R",
    "ankle_R",
    "toe_R",
    "vertebra_C11_extend",
    "vertebra_cervical_1_bend",
    "vertebra_axis_twist",
    "atlas",
    "mandible",
    "scapula_L_supinate",
    "scapula_L_abduct",
    "scapula_L_extend",
    "shoulder_L",
    "shoulder_sup_L",
    "elbow_L",
    "wrist_L",
    "scapula_R_supinate",
    "scapula_R_abduct",
    "scapula_R_extend",
    "shoulder_R",
    "shoulder_sup_R",
    "elbow_R",
    "wrist_R",
    "finger_R",
]
_BODY_NAMES = [
    "torso",
    "pelvis",
    "upper_leg_L",
    "lower_leg_L",
    "foot_L",
    "upper_leg_R",
    "lower_leg_R",
    "foot_R",
    "skull",
    "jaw",
    "scapula_L",
    "upper_arm_L",
    "lower_arm_L",
    "finger_L",
    "scapula_R",
    "upper_arm_R",
    "lower_arm_R",
    "finger_R",
]

_APPENDAGE_NAMES = [
    "foot_L",
    "foot_R",
    "hand_L",
    "hand_R",
    "skull",
]

_END_EFF_NAMES = [
    "foot_L",
    "foot_R",
    "hand_L",
    "hand_R",
]


def _bounded_quat_dist(source: np.ndarray, target: np.ndarray) -> np.ndarray:
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


class RodentTracking(PipelineEnv):
    """Single clip rodent tracking"""

    def __init__(
        self,
        reference_clip,
        torque_actuators: bool = False,
        ref_len: int = 5,
        too_far_dist=0.1,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        ctrl_diff_cost_weight=0.01,
        pos_reward_weight=1.0,
        quat_reward_weight=1.0,
        joint_reward_weight=1.0,
        angvel_reward_weight=1.0,
        bodypos_reward_weight=1.0,
        endeff_reward_weight=1.0,
        healthy_z_range=(0.03, 0.5),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        **kwargs,
    ):
        root = mjcf_dm.from_path(_XML_PATH)

        # Convert to torque actuators
        if torque_actuators:
            for actuator in root.find_all("actuator"):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm

        rescale.rescale_subtree(
            root,
            0.9,
            0.9,
        )
        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations

        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        max_physics_steps_per_control_step = int(
            (1.0 / (_MOCAP_HZ * mj_model.opt.timestep))
        )

        super().__init__(sys, **kwargs)
        if max_physics_steps_per_control_step % physics_steps_per_control_step != 0:
            raise ValueError(
                f"physics_steps_per_control_step ({physics_steps_per_control_step}) must be a factor of ({max_physics_steps_per_control_step})"
            )

        self._steps_for_cur_frame = (
            max_physics_steps_per_control_step / physics_steps_per_control_step
        )
        print(f"self._steps_for_cur_frame: {self._steps_for_cur_frame}")

        self._torso_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), "torso"
        )

        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("joint"), joint)
                for joint in _JOINT_NAMES
            ]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in _BODY_NAMES
            ]
        )

        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in _APPENDAGE_NAMES
            ]
        )

        self._reference_clip = reference_clip
        self._bad_pose_dist = bad_pose_dist
        self._too_far_dist = too_far_dist
        self._bad_quat_dist = bad_quat_dist
        self._ref_len = ref_len
        self._pos_reward_weight = pos_reward_weight
        self._quat_reward_weight = quat_reward_weight
        self._joint_reward_weight = joint_reward_weight
        self._angvel_reward_weight = angvel_reward_weight
        self._bodypos_reward_weight = bodypos_reward_weight
        self._endeff_reward_weight = endeff_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._ctrl_diff_cost_weight = ctrl_diff_cost_weight
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        _, start_rng, rng = jax.random.split(rng, 3)

        start_frame = jax.random.randint(start_rng, (), 0, 44)

        info = {
            "cur_frame": start_frame,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nv,)),
        }

        return self.reset_from_clip(rng, info, noise=True)

    def reset_from_clip(self, rng, info, noise=True) -> State:
        """Reset based on a reference clip."""
        _, rng1, rng2 = jax.random.split(rng, 3)

        # Get reference clip and select the start frame
        reference_frame = jax.tree.map(
            lambda x: x[info["cur_frame"]], self._get_reference_clip(info)
        )

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Add pos
        qpos_with_pos = jp.array(self.sys.qpos0).at[:3].set(reference_frame.position)

        # Add quat
        new_qpos = qpos_with_pos.at[3:7].set(reference_frame.quaternion)

        # Add noise
        qpos = new_qpos + jp.where(
            noise,
            jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi),
            jp.zeros((self.sys.nq,)),
        )

        qvel = jp.where(
            noise,
            jax.random.uniform(rng1, (self.sys.nv,), minval=low, maxval=hi),
            jp.zeros((self.sys.nv,)),
        )

        data = self.pipeline_init(qpos, qvel)

        reference_obs, proprioceptive_obs = self._get_obs(data, info)

        # Used to intialize our intention network
        info["task_obs_size"] = reference_obs.shape[-1]

        obs = jp.concatenate([reference_obs, proprioceptive_obs])

        reward, done, zero = jp.zeros(3)
        metrics = {
            "pos_reward": zero,
            "quat_reward": zero,
            "joint_reward": zero,
            "angvel_reward": zero,
            "bodypos_reward": zero,
            "endeff_reward": zero,
            "reward_ctrlcost": zero,
            "ctrl_diff_cost": zero,
            "too_far": zero,
            "bad_pose": zero,
            "bad_quat": zero,
            "fall": zero,
            "nan": zero,
        }

        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # Logic for moving to next frame to track to maintain timesteps alignment
        # TODO: Update this to just refer to model.timestep
        info = state.info.copy()
        info["steps_taken_cur_frame"] += 1
        info["cur_frame"] += jp.where(
            info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 1, 0
        )
        info["steps_taken_cur_frame"] *= jp.where(
            info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 0, 1
        )

        # Gets reference clip and indexes to current frame
        reference_clip = jax.tree.map(
            lambda x: x[info["cur_frame"]], self._get_reference_clip(info)
        )

        pos_distance = data.qpos[:3] - reference_clip.position
        pos_reward = self._pos_reward_weight * jp.exp(-400 * jp.sum(pos_distance**2))

        quat_distance = jp.sum(
            _bounded_quat_dist(data.qpos[3:7], reference_clip.quaternion) ** 2
        )
        quat_reward = self._quat_reward_weight * jp.exp(-4.0 * quat_distance)

        joint_distance = jp.sum((data.qpos[7:] - reference_clip.joints) ** 2)
        joint_reward = self._joint_reward_weight * jp.exp(-0.25 * joint_distance)
        info["joint_distance"] = joint_distance

        angvel_reward = self._angvel_reward_weight * jp.exp(
            -0.5 * jp.sum((data.qvel[3:6] - reference_clip.angular_velocity) ** 2)
        )

        bodypos_reward = self._bodypos_reward_weight * jp.exp(
            -8.0
            * jp.sum(
                (
                    data.xpos[self._body_idxs]
                    - reference_clip.body_positions[self._body_idxs]
                ).flatten()
                ** 2
            )
        )

        endeff_reward = self._endeff_reward_weight * jp.exp(
            -500
            * jp.sum(
                (
                    data.xpos[self._endeff_idxs]
                    - reference_clip.body_positions[self._endeff_idxs]
                ).flatten()
                ** 2
            )
        )

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._torso_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._torso_idx][2] > max_z, 0.0, is_healthy)
        fall = 1.0 - is_healthy

        summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
        too_far = jp.where(summed_pos_distance > self._too_far_dist, 1.0, 0.0)
        info["summed_pos_distance"] = summed_pos_distance
        info["quat_distance"] = quat_distance
        bad_pose = jp.where(joint_distance > self._bad_pose_dist, 1.0, 0.0)
        bad_quat = jp.where(quat_distance > self._bad_quat_dist, 1.0, 0.0)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        ctrl_diff_cost = self._ctrl_diff_cost_weight * jp.sum(
            jp.square(info["prev_ctrl"] - action)
        )
        info["prev_ctrl"] = action
        reference_obs, proprioceptive_obs = self._get_obs(data, info)
        obs = jp.concatenate([reference_obs, proprioceptive_obs])
        reward = (
            joint_reward
            + pos_reward
            + quat_reward
            + angvel_reward
            + bodypos_reward
            + endeff_reward
            - ctrl_cost
            - ctrl_diff_cost
        )

        # Raise done flag if terminating
        done = jp.max(jp.array([fall, too_far, bad_pose, bad_quat]))

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max(jp.array([nan, done]))

        state.metrics.update(
            pos_reward=pos_reward,
            quat_reward=quat_reward,
            joint_reward=joint_reward,
            angvel_reward=angvel_reward,
            bodypos_reward=bodypos_reward,
            endeff_reward=endeff_reward,
            reward_ctrlcost=-ctrl_cost,
            ctrl_diff_cost=-ctrl_diff_cost,
            too_far=too_far,
            bad_pose=bad_pose,
            bad_quat=bad_quat,
            fall=fall,
            nan=nan,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_reference_clip(self, info) -> ReferenceClip:
        """Returns reference clip; to be overridden in child classes"""
        return self._reference_clip

    def _get_reference_trajectory(self, info) -> ReferenceClip:
        """Slices ReferenceClip into the observation trajectory"""

        # Get the relevant slice of the reference clip
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    info["cur_frame"] + 1,
                    self._ref_len,
                )
            return jp.array([])

        return jax.tree.map(f, self._get_reference_clip(info))

    def _get_obs(self, data: mjx.Data, info) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""

        ref_traj = self._get_reference_trajectory(info)

        track_pos_local = jax.vmap(
            lambda a, b: brax_math.rotate(a, b), in_axes=(0, None)
        )(
            ref_traj.position - data.qpos[:3],
            data.qpos[3:7],
        ).flatten()

        quat_dist = jax.vmap(
            lambda a, b: brax_math.relative_quat(a, b), in_axes=(0, None)
        )(
            ref_traj.quaternion,
            data.qpos[3:7],
        ).flatten()

        joint_dist = (ref_traj.joints - data.qpos[7:])[:, self._joint_idxs].flatten()

        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(brax_math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (ref_traj.body_positions - data.xpos)[:, self._body_idxs],
            data.qpos[3:7],
        ).flatten()

        reference_obs = jp.concatenate(
            [
                track_pos_local,
                quat_dist,
                joint_dist,
                body_pos_dist_local,
            ]
        )

        prorioceptive_obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
            ]
        )
        return reference_obs, prorioceptive_obs


class RodentMultiClipTracking(RodentTracking):
    def __init__(
        self,
        reference_clip,
        torque_actuators: bool = False,
        ref_len: int = 5,
        too_far_dist=0.1,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        ctrl_diff_cost_weight=0.01,
        pos_reward_weight=1,
        quat_reward_weight=1,
        joint_reward_weight=1,
        angvel_reward_weight=1,
        bodypos_reward_weight=1,
        endeff_reward_weight=1,
        healthy_z_range=(0.03, 0.5),
        physics_steps_per_control_step=10,
        reset_noise_scale=0.001,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        **kwargs,
    ):
        super().__init__(
            None,
            torque_actuators,
            ref_len,
            too_far_dist,
            bad_pose_dist,
            bad_quat_dist,
            ctrl_cost_weight,
            ctrl_diff_cost_weight,
            pos_reward_weight,
            quat_reward_weight,
            joint_reward_weight,
            angvel_reward_weight,
            bodypos_reward_weight,
            endeff_reward_weight,
            healthy_z_range,
            physics_steps_per_control_step,
            reset_noise_scale,
            solver,
            iterations,
            ls_iterations,
            **kwargs,
        )

        self._reference_clips = reference_clip
        self._n_clips = reference_clip.position.shape[0]

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        _, start_rng, clip_rng, rng = jax.random.split(rng, 4)

        start_frame = jax.random.randint(start_rng, (), 0, 44)
        clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)
        info = {
            "clip_idx": clip_idx,
            "cur_frame": start_frame,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info, noise=True)

    def _get_reference_clip(self, info) -> ReferenceClip:
        """Gets clip based on info["clip_idx"]"""

        return jax.tree.map(lambda x: x[info["clip_idx"]], self._reference_clips)


def get_config():
    """Returns reward config for barkour quadruped environment."""

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                # The coefficients for all reward terms used for training. All
                # physical quantities are in SI units, if no otherwise specified,
                # i.e. joint positions are in rad, positions are measured in meters,
                # torques in Nm, and time in seconds, and forces in Newtons.
                scales=config_dict.ConfigDict(
                    dict(
                        # Tracking rewards are computed using exp(-delta^2/sigma)
                        # sigma can be a hyperparameters to tune.
                        # Track the base x-y velocity (no z-velocity tracking.)
                        tracking_lin_vel=150.0,
                        # Track the angular velocity along z-axis, i.e. yaw rate.
                        tracking_ang_vel=80.0,
                        # Below are regularization terms, we roughly divide the
                        # terms to base state regularizations, joint
                        # regularizations, and other behavior regularizations.
                        # Penalize the base velocity in z direction, L2 penalty.
                        lin_vel_z=-0.0002,
                        # Penalize the base roll and pitch rate. L2 penalty.
                        ang_vel_xy=-0.0005,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=-0.5,
                        # L2 regularization of joint torques, |tau|^2.
                        torques=-0.0002,
                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=-0.01,
                        # Encourage long swing steps.  However, it does not
                        # encourage high clearances.
                        # feet_air_time=0.2,
                        # Encourage no motion at zero command, L2 regularization
                        # |q - q_default|^2.
                        stand_still=-0.05,
                        # Early termination penalty.
                        termination=-1.0,
                        # Penalizing foot slipping on the ground.
                        # foot_slip=-0.1,
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.0025,
            )
        )
        return default_config

    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )

    return default_config


class RodentJoystick(PipelineEnv):
    """Environment for training the barkour quadruped joystick policy in MJX."""

    def __init__(
        self,
        obs_noise: float = 0.001,
        action_scale: float = 1.0,
        kick_vel: float = 0.005,
        torque_actuators: bool = False,
        solver: str = "cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        **kwargs,
    ):
        root = mjcf_dm.from_path(_XML_PATH)

        # Convert to torque actuators
        if torque_actuators:
            for actuator in root.find_all("actuator"):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm

        rescale.rescale_subtree(
            root,
            0.9,
            0.9,
        )
        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations

        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        self._dt = 0.01  # Pretrained environment is 100 fps

        # sys = sys.tree_replace({"opt.timestep": 0.004})

        n_frames = kwargs.pop("n_frames", int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self.reward_config = get_config()
        # set custom from kwargs
        for k, v in kwargs.items():
            if k.endswith("_scale"):
                self.reward_config.rewards.scales[k[:-6]] = v

        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "torso"
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(sys.mj_model.qpos0)
        self._default_pose = sys.mj_model.qpos0[7:]
        # self.lowers = jp.array([-0.7, -1.0, 0.05] * 4)
        # self.uppers = jp.array([0.52, 2.1, 2.1] * 4)
        # feet_site_id = [
        #     mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
        #     for f in _END_EFF_NAMES
        # ]
        # assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        # self._feet_site_id = np.array(feet_site_id)
        # lower_leg_body = [
        #     "lower_leg_front_left",
        #     "lower_leg_hind_left",
        #     "lower_leg_front_right",
        #     "lower_leg_hind_right",
        # ]
        # lower_leg_body_id = [
        #     mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
        #     for l in lower_leg_body
        # ]
        # assert not any(id_ == -1 for id_ in lower_leg_body_id), "Body not found."
        # self._lower_leg_body_id = np.array(lower_leg_body_id)
        # self._foot_radius = 0.003210712207833968
        self._nv = sys.nv

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [-0.1, 0.3]  # min max [m/s]
        lin_vel_y = [-0.15, 0.15]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(self.sys.nu),
            "last_vel": jp.zeros(self.sys.nv - 6),
            "command": self.sample_command(key),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4),
            "rewards": {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
            "kick": jp.array([0.0, 0.0]),
            "step": 0,
        }

        # obs_history = jp.zeros(15 * 31)  # store 15 steps of history
        # obs = self._get_obs(pipeline_state, state_info, obs_history)
        task_obs, proprioceptive_obs = self._get_obs(pipeline_state, state_info)

        state_info["task_obs_size"] = task_obs.shape[-1]

        obs = jp.concatenate([task_obs, proprioceptive_obs])
        reward, done = jp.zeros(2)
        metrics = {
            "total_dist": 0.0,
            "nan": 0.0,
            "fall": 0.0,
            "flip": 0.0,
        }
        for k in state_info["rewards"]:
            metrics[k] = state_info["rewards"][k]
        state = State(
            pipeline_state, obs, reward, done, metrics, state_info
        )  # pytype: disable=wrong-arg-types
        return state

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info["rng"], 3)

        # kick
        # push_interval = 10
        # kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
        # kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        # kick *= jp.mod(state.info["step"], push_interval) == 0
        # qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        # qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        # state = state.tree_replace({"pipeline_state.qvel": qvel})

        # physics step
        # motor_targets = self._default_pose + action * self._action_scale
        # motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        task_obs, proprioceptive_obs = self._get_obs(
            pipeline_state, state.info  # , state.obs
        )
        obs = jp.concatenate([task_obs, proprioceptive_obs])
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        # foot_pos = pipeline_state.site_xpos[
        #     self._feet_site_id
        # ]  # pytype: disable=attribute-error
        # foot_contact_z = foot_pos[:, 2] - self._foot_radius
        # contact = foot_contact_z < 1e-3  # a mm or less off the floor
        # contact_filt_mm = contact | state.info["last_contact"]
        # contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        # first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        # state.info["feet_air_time"] += self.dt

        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        flip = jp.dot(brax_math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        # done |= jp.any(joint_angles < self.lowers)
        # done |= jp.any(joint_angles > self.uppers)

        # Modified fall value for rodent
        fall = pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.0325
        done = fall | flip

        # reward
        rewards = {
            "tracking_lin_vel": (
                self._reward_tracking_lin_vel(state.info["command"], x, xd)
            ),
            "tracking_ang_vel": (
                self._reward_tracking_ang_vel(state.info["command"], x, xd)
            ),
            "lin_vel_z": self._reward_lin_vel_z(xd),
            "ang_vel_xy": self._reward_ang_vel_xy(xd),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(
                pipeline_state.qfrc_actuator
            ),  # pytype: disable=attribute-error
            "action_rate": self._reward_action_rate(action, state.info["last_act"]),
            "stand_still": self._reward_stand_still(
                state.info["command"],
                joint_angles,
            ),
            # "feet_air_time": self._reward_feet_air_time(
            #     state.info["feet_air_time"],
            #     first_contact,
            #     state.info["command"],
            # ),
            # "foot_slip": self._reward_foot_slip(pipeline_state, contact_filt_cm),
            "termination": self._reward_termination(done, state.info["step"]),
        }
        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(pipeline_state)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, True, False)
        done |= nan

        state.metrics["nan"] = jp.float32(nan)
        state.metrics["fall"] = jp.float32(fall)
        state.metrics["flip"] = jp.float32(flip)
        # state management
        # state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        # state.info["feet_air_time"] *= ~contact_filt_mm
        # state.info["last_contact"] = contact
        state.info["rewards"] = rewards
        state.info["step"] += 1
        state.info["rng"] = rng

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jp.where(
            state.info["step"] > 250,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        # reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > 250), 0, state.info["step"]
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = brax_math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info["rewards"])

        done = jp.float32(done)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        # obs_history: jax.Array,
    ) -> jax.Array:
        inv_torso_rot = brax_math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = brax_math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        task_obs = jp.concatenate(
            [
                jp.array([local_rpyrate[2]]) * 0.25,  # yaw rate
                brax_math.rotate(
                    jp.array([0, 0, -1]), inv_torso_rot
                ),  # projected gravity
                state_info["command"] * jp.array([2.0, 2.0, 0.25]),  # command
                # pipeline_state.q[7:] - self._default_pose,  # motor angles
                state_info["last_act"],  # last action
            ]
        )

        prorioceptive_obs = jp.concatenate(
            [
                pipeline_state.qpos,
                pipeline_state.qvel,
            ]
        )

        # clip, noise
        task_obs = jp.clip(
            task_obs, -100.0, 100.0
        ) + self._obs_noise * jax.random.uniform(
            state_info["rng"], task_obs.shape, minval=-1, maxval=1
        )
        # stack observations through time
        # obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return task_obs, prorioceptive_obs

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = brax_math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jp.sum(jp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = brax_math.rotate(xd.vel[0], brax_math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(
            -lin_vel_error / self.reward_config.rewards.tracking_sigma
        )
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = brax_math.rotate(xd.ang[0], brax_math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (
            brax_math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
            brax_math.normalize(commands[:2])[1] < 0.1
        )

    # def _reward_foot_slip(
    #     self, pipeline_state: base.State, contact_filt: jax.Array
    # ) -> jax.Array:
    #     # get velocities at feet which are offset from lower legs
    #     # pytype: disable=attribute-error
    #     pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
    #     feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
    #     # pytype: enable=attribute-error
    #     offset = base.Transform.create(pos=feet_offset)
    #     foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
    #     foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

    #     # Penalize large feet velocity for feet that are in contact with the ground.
    #     return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 200)

    # def render(
    #     self,
    #     trajectory: List[base.State],
    #     camera: str | None = None,
    #     width: int = 240,
    #     height: int = 320,
    # ) -> Sequence[np.ndarray]:
    #     camera = camera or "track"
    #     return super().render(trajectory, camera=camera, width=width, height=height)


class RodentRun(PipelineEnv):

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.0325, 0.1),
        reset_noise_scale=1e-3,
        exclude_current_positions_from_observation=True,
        torque_actuators: bool = False,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        **kwargs,
    ):
        root = mjcf_dm.from_path(_XML_PATH)

        # Convert to torque actuators
        if torque_actuators:
            for actuator in root.find_all("actuator"):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm

        rescale.rescale_subtree(
            root,
            0.9,
            0.9,
        )
        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations

        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", 5)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self._torso_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), "torso"
        )

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        task_obs, proprioceptive_obs = self._get_obs(data, jp.zeros(self.sys.nu))
        obs = jp.concatenate([task_obs, proprioceptive_obs])
        info = {"task_obs_size": task_obs.size[-1]}

        reward, done, zero = jp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "nan": zero,
            "fall": zero,
        }
        return State(data, obs, reward, done, metrics, info=info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._torso_idx] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._torso_idx] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        task_obs, proprioceptive_obs = self._get_obs(data, action)
        obs = jp.concatenate([task_obs, proprioceptive_obs])
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max([nan, done])

        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.linalg.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            nan=nan,
            fall=1.0 - is_healthy,
        )

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        task_obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        )

        proprioceptive_obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
            ]
        )

        return task_obs, proprioceptive_obs
