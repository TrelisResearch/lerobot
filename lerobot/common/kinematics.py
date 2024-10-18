# ruff: noqa: N806, N815, N803
import numpy as np
from scipy.spatial.transform import Rotation


def skew_symmetric(w):
    """Creates the skew-symmetric matrix from a 3D vector."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rodrigues_rotation(w, theta):
    """Computes the rotation matrix using Rodrigues' formula."""
    w_hat = skew_symmetric(w)
    return np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat


def screw_axis_to_transform(S, theta):
    """Converts a screw axis to a 4x4 transformation matrix."""
    S_w = S[:3]
    S_v = S[3:]
    if np.allclose(S_w, 0) and np.linalg.norm(S_v) == 1:  # Pure translation
        T = np.eye(4)
        T[:3, 3] = S_v * theta
    elif np.linalg.norm(S_w) == 1:  # Rotation and translation
        w_hat = skew_symmetric(S_w)
        R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat
        t = (np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * w_hat @ w_hat) @ S_v
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
    else:
        raise ValueError("Invalid screw axis parameters")
    return T


def pose_difference_se3(pose1, pose2):
    """
    Calculates the SE(3) difference between two 4x4 homogeneous transformation matrices.

    pose1 - pose2

    Args:
        pose1: A 4x4 numpy array representing the first pose.
        pose2: A 4x4 numpy array representing the second pose.

    Returns:
        A tuple (translation_diff, rotation_diff) where:
        - translation_diff is a 3x1 numpy array representing the translational difference.
        - rotation_diff is a 3x1 numpy array representing the rotational difference in axis-angle representation.
    """

    # Extract rotation matrices from poses
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]

    # Calculate translational difference
    translation_diff = pose1[:3, 3] - pose2[:3, 3]

    # Calculate rotational difference using scipy's Rotation library
    R_diff = Rotation.from_matrix(R1 @ R2.T)
    rotation_diff = R_diff.as_rotvec()  # Convert to axis-angle representation

    return np.concatenate([translation_diff, rotation_diff])


def se3_error(target_pose, current_pose):
    pos_error = target_pose[:3, 3] - current_pose[:3, 3]
    R_target = target_pose[:3, :3]
    R_current = current_pose[:3, :3]
    R_error = R_target @ R_current.T
    rot_error = Rotation.from_matrix(R_error).as_rotvec()
    return np.concatenate([pos_error, rot_error])


class KochKinematics:
    gripper_X0 = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    # Screw axis of gripper frame wrt base frame.
    S_BG = np.array([1, 0, 0, 0, 0.024, 0.001])
    # Gripper origin to centroid transform.
    X_GoGc = np.array(
        [
            [1, 0, 0, 0.048],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # Gripper origin to tip transform.
    X_GoGt = np.array(
        [
            [1, 0, 0, 0.083],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # 0-position gripper frame pose wrt base.
    X_BoGo = np.array(
        [
            [1, 0, 0, 0.239],
            [0, 1, 0, -0.001],
            [0, 0, 1, 0.024],
            [0, 0, 0, 1],
        ]
    )

    # Wrist
    wrist_X0 = np.array(
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Screw axis of wrist frame wrt base frame.
    S_BR = np.array([0, 1, 0, -0.024, 0, 0.209])
    # 0-position origin to centroid transform.
    X_RoRc = np.array(
        [
            [1, 0, 0, 0.0035],
            [0, 1, 0, -0.002],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # 0-position wrist frame pose wrt base.
    X_BR = np.array(
        [
            [1, 0, 0, 0.209],
            [0, 1, 0, 0],
            [0, 0, 1, 0.024],
            [0, 0, 0, 1],
        ]
    )

    # Screw axis of forearm frame wrt base frame.
    S_BF = np.array([0, 1, 0, -0.020, 0, 0.108])
    # Forearm origin + centroid transform.
    X_FoFc = np.array(
        [
            [1, 0, 0, 0.036],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # 0-position forearm frame pose wrt base.
    X_BF = np.array(
        [
            [1, 0, 0, 0.108],
            [0, 1, 0, 0],
            [0, 0, 1, 0.020],
            [0, 0, 0, 1],
        ]
    )

    # Screw axis of humerus frame wrt base frame.
    S_BH = np.array([0, -1, 0, 0.036, 0, 0])
    # Humerus origin to centroid transform.
    X_HoHc = np.array(
        [
            [1, 0, 0, 0.0475],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # 0-position humerus frame pose wrt base.
    X_BH = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.036],
            [0, 0, 0, 1],
        ]
    )

    # Screw axis of shoulder frame wrt Base frame.
    S_BS = np.array([0, 0, -1, 0, 0, 0])
    X_SoSc = np.array(
        [
            [1, 0, 0, -0.017],
            [0, 1, 0, 0],
            [0, 0, 1, 0.0235],
            [0, 0, 0, 1],
        ]
    )
    X_BS = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1],
        ]
    )

    # o3d seems to be aligning the box frame so that it is longest to longest to shortest on xyz.
    base_X0 = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    X_BoBc = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    X_WoBo = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1],
        ]
    )

    _fk_gripper_post = X_GoGc @ X_BoGo @ gripper_X0

    @staticmethod
    def fk_base():
        return KochKinematics.X_WoBo @ KochKinematics.X_BoBc @ KochKinematics.base_X0

    @staticmethod
    def fk_shoulder(robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            KochKinematics.X_WoBo
            @ screw_axis_to_transform(KochKinematics.S_BS, robot_pos_rad[0])
            @ KochKinematics.X_SoSc
            @ KochKinematics.X_BS
        )

    @staticmethod
    def fk_humerus(robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            KochKinematics.X_WoBo
            @ screw_axis_to_transform(KochKinematics.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(KochKinematics.S_BH, robot_pos_rad[1])
            @ KochKinematics.X_HoHc
            @ KochKinematics.X_BH
        )

    @staticmethod
    def fk_forearm(robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            KochKinematics.X_WoBo
            @ screw_axis_to_transform(KochKinematics.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(KochKinematics.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(KochKinematics.S_BF, robot_pos_rad[2])
            @ KochKinematics.X_FoFc
            @ KochKinematics.X_BF
        )

    @staticmethod
    def fk_wrist(robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            KochKinematics.X_WoBo
            @ screw_axis_to_transform(KochKinematics.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(KochKinematics.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(KochKinematics.S_BF, robot_pos_rad[2])
            @ screw_axis_to_transform(KochKinematics.S_BR, robot_pos_rad[3])
            @ KochKinematics.X_RoRc
            @ KochKinematics.X_BR
            @ KochKinematics.wrist_X0
        )

    @staticmethod
    def fk_gripper(robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            KochKinematics.X_WoBo
            @ screw_axis_to_transform(KochKinematics.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(KochKinematics.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(KochKinematics.S_BF, robot_pos_rad[2])
            @ screw_axis_to_transform(KochKinematics.S_BR, robot_pos_rad[3])
            @ screw_axis_to_transform(KochKinematics.S_BG, robot_pos_rad[4])
            @ KochKinematics._fk_gripper_post
        )

    @staticmethod
    def fk_gripper_tip(robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            KochKinematics.X_WoBo
            @ screw_axis_to_transform(KochKinematics.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(KochKinematics.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(KochKinematics.S_BF, robot_pos_rad[2])
            @ screw_axis_to_transform(KochKinematics.S_BR, robot_pos_rad[3])
            @ screw_axis_to_transform(KochKinematics.S_BG, robot_pos_rad[4])
            @ KochKinematics.X_GoGt
            @ KochKinematics.X_BoGo
            @ KochKinematics.gripper_X0
        )

    @staticmethod
    def jac_gripper(robot_pos_deg):
        """Finite differences to compute the Jacobian.
        J(i, j) represents how the ith component of the end-effector's velocity changes wrt a small change
        in the jth joint's velocity.
        """
        eps = 1e-8
        jac = np.zeros(shape=(6, 5))
        delta = np.zeros(len(robot_pos_deg[:-1]), dtype=np.float64)
        for el_ix in range(len(robot_pos_deg[:-1])):
            delta *= 0
            delta[el_ix] = eps / 2
            Sdot = (
                pose_difference_se3(
                    KochKinematics.fk_gripper(robot_pos_deg[:-1] + delta),
                    KochKinematics.fk_gripper(robot_pos_deg[:-1] - delta),
                )
                / eps
            )
            jac[:, el_ix] = Sdot
        return jac

    @staticmethod
    def pos_jac_gripper(robot_pos_deg):
        """Finite differences to compute the Jacobian.
        J(i, j) represents how the ith component of the end-effector's velocity changes wrt a small change
        in the jth joint's velocity.
        """
        eps = 1e-8
        jac = np.zeros(shape=(3, 5))
        delta = np.zeros(len(robot_pos_deg[:-1]), dtype=np.float64)
        for el_ix in range(len(robot_pos_deg[:-1])):
            delta *= 0
            delta[el_ix] = eps / 2
            Sdot = (
                KochKinematics.fk_gripper(robot_pos_deg[:-1] + delta)[:3, 3]
                - KochKinematics.fk_gripper(robot_pos_deg[:-1] - delta)[:3, 3]
            ) / eps
            jac[:, el_ix] = Sdot
        return jac

    @staticmethod
    def ik(current_joint_state, desired_ee_pose, position_only=True):
        # Do gradient descent.
        max_iterations = 5
        learning_rate = 1
        for _ in range(max_iterations):
            current_ee_pose = KochKinematics.fk_gripper(current_joint_state)
            if not position_only:
                error = se3_error(desired_ee_pose, current_ee_pose)
                jac = KochKinematics.jac_gripper(current_joint_state)
            else:
                error = desired_ee_pose[:3, 3] - current_ee_pose[:3, 3]
                jac = KochKinematics.pos_jac_gripper(current_joint_state)
            delta_angles = np.linalg.pinv(jac) @ error
            current_joint_state[:-1] += learning_rate * delta_angles

            if np.linalg.norm(error) < 5e-3:
                return current_joint_state
        return current_joint_state
