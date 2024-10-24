# ruff: noqa: N803, N806
import logging

import numpy as np
import open3d as o3d

from lerobot.common.kinematics import RobotKinematics

GREEN = np.array([0.5, 1.0, 0.5])
BLUE = np.array([0.0, 0.0, 1.0])
RED = np.array([1.0, 0.1, 0.1])
PURPLE = np.array([0.8, 0, 0.8])
LIGHT_GRAY = np.array([0.8, 0.8, 0.8])


class DigitalTwin:
    def __init__(self, window_name: str = "Digital Twin"):
        # Frames. All origins of robot link frames are at the center of the respective motor axis.
        # (W)orld
        # (B)ase # Bo is at the center of the motor axis of the shoulder pan motor.
        # (S)houlder  # So is at the point where the shoulder pan motor connects to the shoulder lift motor
        # (H)umerus  # Ho is on the center of the shoulder lift motor axis
        # (F)orearm  # Fo is on the center of the elbow lift motor axis
        #  w(R)ist  # Wo is on the center of the wrist lift motor axis
        # (G)ripper  # Go is at the point where the wrist twist motor connects to the gripper

        # Gripper
        self.gripper = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0],
            R=np.eye(3),
            extent=[0.083, 0.036, 0.035],  # (0.083, 0.035, 0.036)
        )
        # self.gripper.compute_vertex_normals()
        # self.gripper.paint_uniform_color(GREEN)
        self.gripper.color = GREEN

        self.wrist = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(
                center=[0, 0, 0],
                R=np.eye(3),
                extent=[0.042, 0.027, 0.02],  # (0.025, 0.042, 0.02)
            )
        )
        self.wrist.compute_vertex_normals()
        self.wrist.paint_uniform_color(GREEN)

        # Forearm
        self.forearm = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(
                center=[0, 0, 0],
                R=np.eye(3),
                extent=[0.09, 0.035, 0.024],
            )
        )
        self.forearm.compute_vertex_normals()
        self.forearm.paint_uniform_color(GREEN)

        # Humerus
        self.humerus = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(
                center=[0, 0, 0],
                R=np.eye(3),
                extent=[0.125, 0.045, 0.025],
            )
        )
        self.humerus.compute_vertex_normals()
        self.humerus.paint_uniform_color(GREEN)

        # Shoulder
        self.shoulder = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(
                center=[0, 0, 0],
                R=np.eye(3),
                extent=[0.05, 0.04, 0.007],  # (0.035, 0.05, 0.04)
            )
        )
        self.shoulder.compute_vertex_normals()
        self.shoulder.paint_uniform_color(GREEN)

        # Base to world transform.
        self.base = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(
                center=[0, 0, 0],
                R=np.eye(3),
                extent=[0.05, 0.04, 0.035],  # (0.035, 0.05, 0.04)
            )
        )
        self.base.compute_vertex_normals()
        self.base.paint_uniform_color(GREEN)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # Create a floor
        self.floor = o3d.geometry.TriangleMesh.create_box(width=10.0, height=10.0, depth=0.001)  # x, y, z
        self.floor.translate([-5.0, -5.0, -0.001])
        self.floor.compute_vertex_normals()
        self.floor.paint_uniform_color(LIGHT_GRAY)

        # Grid parameters
        grid_size = 10.0  # Size of the grid in each dimension
        grid_spacing = 0.1  # 10 cm spacing

        # Create grid lines
        lines = []
        colors = [[0, 0, 0] for _ in range(int(2 * grid_size / grid_spacing))]  # Black color for all lines

        # Lines along X-axis
        for i in range(int(grid_size / grid_spacing) + 1):
            x = -grid_size / 2 + i * grid_spacing
            lines.append([[-grid_size / 2, x, 0], [grid_size / 2, x, 0]])

        # Lines along Y-axis
        for i in range(int(grid_size / grid_spacing) + 1):
            y = -grid_size / 2 + i * grid_spacing
            lines.append([[y, -grid_size / 2, 0], [y, grid_size / 2, 0]])

        # Create LineSet geometry
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3)),
            lines=o3d.utility.Vector2iVector(np.arange(len(lines) * 2).reshape(-1, 2)),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Create a visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=window_name, width=768, height=512)
        self.vis.add_geometry(self.floor)
        self.vis.add_geometry(self.base)
        self.vis.add_geometry(self.shoulder)
        self.vis.add_geometry(self.humerus)
        self.vis.add_geometry(self.forearm)
        self.vis.add_geometry(self.wrist)
        self.vis.add_geometry(self.gripper)
        self.vis.add_geometry(coordinate_frame)
        self.vis.add_geometry(line_set)

        self._do_quit = False
        self.vis.register_key_callback(ord("Q"), self.set_quit)

        # Hide a bunch of waypoints under the plane.
        self.waypoints = [o3d.geometry.TriangleMesh.create_tetrahedron(radius=0.005) for i in range(1000)]
        for waypoint in self.waypoints:
            waypoint.translate([0, 0, -0.1])
            waypoint.compute_vertex_normals()
            waypoint.paint_uniform_color(PURPLE)
            self.vis.add_geometry(waypoint)

        # Set initial view control
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.025)
        view_control.set_up([0, 0, 1])
        view_control.set_front([0.2, -1, 0.3])  # Set camera looking towards the cube
        view_control.set_lookat([0, 0, 0.05])  # Set camera focus point

    def set_quit(self, *_):
        self._do_quit = True

    def quit_signal_is_set(self):
        return self._do_quit

    def set_object_pose(self, obj, X_WO):
        if obj is self.gripper:
            self.gripper.center = X_WO[:3, 3]
            self.gripper.R = X_WO[:3, :3]
            return
        box = obj.get_oriented_bounding_box()
        obj.translate(-box.center)
        obj.rotate(np.linalg.inv(box.R))
        obj.transform(X_WO)

    def set_twin_pose(self, follower_pos, follower_pos_trajectory=None):
        # follower_pos *= 0
        self.set_object_pose(self.base, RobotKinematics.fk_base())
        self.vis.update_geometry(self.base)
        self.set_object_pose(self.shoulder, RobotKinematics.fk_shoulder(follower_pos))
        self.vis.update_geometry(self.shoulder)
        self.set_object_pose(self.humerus, RobotKinematics.fk_humerus(follower_pos))
        self.vis.update_geometry(self.humerus)
        self.set_object_pose(self.forearm, RobotKinematics.fk_forearm(follower_pos))
        self.vis.update_geometry(self.forearm)
        self.set_object_pose(self.wrist, RobotKinematics.fk_wrist(follower_pos))
        self.vis.update_geometry(self.wrist)
        self.set_object_pose(self.gripper, RobotKinematics.fk_gripper(follower_pos))
        # self.gripper.paint_uniform_color(
        #     np.clip((1 - follower_pos[-1] / 50) * RED + (follower_pos[-1] / 50) * GREEN, 0, 1)
        # )
        self.gripper.color = np.clip(
            (1 - follower_pos[-1] / 50) * RED + (follower_pos[-1] / 50) * GREEN, 0, 1
        )
        self.vis.update_geometry(self.gripper)

        if follower_pos_trajectory is not None:
            self._set_gripper_waypoints(follower_pos_trajectory)

        # Update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()

    def _set_gripper_waypoints(self, follower_pos_trajectory):
        if follower_pos_trajectory.shape[0] > len(self.waypoints):
            logging.warning("Not enough waypoint objects loaded into the scene to show the full trajectory.")
        # Rest all waypoints to below the floor.
        for waypoint in self.waypoints:
            waypoint.translate([0, 0, -0.1], relative=False)
        # Set the necessary number of waypoints.
        for i, (waypoint, follower_pos) in enumerate(
            zip(self.waypoints, follower_pos_trajectory, strict=False)
        ):
            pos = RobotKinematics.fk_gripper(follower_pos)[:3, 3]
            waypoint.translate(pos, relative=False)
            self.vis.update_geometry(waypoint)
            waypoint.paint_uniform_color(
                np.clip(
                    (1 - i / len(follower_pos_trajectory)) * BLUE + (i / len(follower_pos_trajectory)) * RED,
                    0,
                    1,
                )
            )

    def close(self):
        self.vis.destroy_window()

    def __del__(self):
        self.close()
