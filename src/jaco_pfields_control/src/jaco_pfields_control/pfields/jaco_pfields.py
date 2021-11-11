"""
Library for the Modulation of Jaco Pfields
"""
# Author: Deepak Gopinath
# Email: deepakgopinathmusic@gmail.com
# License: BSD (c) 2021

import rospy
import rospy.rostime as rostime
import numpy as np
import tf.transformations as tfs
import sys
import math
import tf2_ros
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

npa = np.array


class JacoPfields(object):
    def __init__(self):
        # attractor position
        # attractor orientation (quat)
        # obstacles position (list of 3D)
        # obstacles orientations (list of 4D quats)
        self.goal_position = npa([0] * 3, dtype="f")
        self.goal_quat = npa([0] * 4, dtype="f")
        self.num_dof = 9

        # self.buffer = tf2_ros.Buffer()  # needed for self.collision and table checks. For any pfield that uses jaco

        self.pfieldvel_array = npa([[0] * 3], dtype="f")
        self.pfieldrot_array = npa([[0] * 3], dtype="f")  # the rotational part of twist
        self.repel_sum = npa([0] * 3, dtype="f")  # repelling velocity only to affect the translations.

        # params for jaco computation
        self.cylinder_rad = 0.3
        self.cylinder_h = 0.4
        self.theta_to_goal = 0.0

        self.istransdone = False
        self.isrotdone = False
        self.iswithintable = False
        self.iswithinobs = False
        self.iswithinbase = False

        self.model_source_frameid = "j2s7s300_link_base"

        self.eef_position = npa([0] * 3, dtype="f")
        self.eef_quat = npa([0] * 4, dtype="f")
        self.diff_quat = npa([0] * 4, dtype="f")
        self.hand_position = npa([0] * 3, dtype="f")
        self.finger_1_position = npa([0] * 3, dtype="f")
        self.finger_2_position = npa([0] * 3, dtype="f")
        self.finger_3_position = npa([0] * 3, dtype="f")

        self.num_obstacles = 0  # update in setter? or pass as arg
        self.obstacles_position = None

        self.dist_to_obs = None
        self.obs_threshold = 0.12  # to be tuned properly

    def set_goal_position_and_quat(self, goal_position, goal_quat):
        # goal_position is geomVector3
        # goal_quat is geometry_msgs/Quaternion
        self.goal_position = npa([goal_position.x, goal_position.y, goal_position.z])
        self.goal_quat = npa([goal_quat.x, goal_quat.y, goal_quat.z, goal_quat.w])
        return True

    def set_obstacles_position_and_quat(self, obsdescs_list):
        # obsdescs_list is a list of obsdescs
        self.num_obstacles = len(obsdescs_list)
        self.dist_to_obs = [0] * self.num_obstacles
        self.obstacles_position = npa([[0] * 3] * self.num_obstacles, dtype="f")
        for i in range(self.num_obstacles):
            obs_position = obsdescs_list[i].position  # Vector3
            self.obstacles_position[i][0] = obs_position.x
            self.obstacles_position[i][1] = obs_position.y
            self.obstacles_position[i][2] = obs_position.z

        # ignore quat for obstacles

    def update_current_pose(self, position, orientation, hand, finger_1, finger_2, finger_3):
        # position - Vector3
        # orientation - Quaternion
        self.eef_position = npa([position.x, position.y, position.z])
        self.eef_quat = npa([orientation.x, orientation.y, orientation.z, orientation.w])
        self.hand_position = npa([hand.x, hand.y, hand.z])
        self.finger_1_position = npa([finger_1.x, finger_1.y, finger_1.z])
        self.finger_2_position = npa([finger_2.x, finger_2.y, finger_2.z])
        self.finger_3_position = npa([finger_3.x, finger_3.y, finger_3.z])

    def reset_flags(self):
        self.istransdone = False
        self.isrotdone = False

    def compute_pfield_vel(
        self,
        robot_position,
        robot_quat,
        robot_hand_position,
        robot_finger_position_1,
        robot_finger_position_2,
        robot_finger_position_3,
    ):
        self.update_current_pose(
            robot_position,
            robot_quat,
            robot_hand_position,
            robot_finger_position_1,
            robot_finger_position_2,
            robot_finger_position_3,
        )
        self.reset_flags()
        # pass current position and orientation of robot as argument
        self.update_attractor_vels()
        self.update_attractor_rotvels()

        # #repeller stuff
        self.update_repeller_vels()
        # self.update_base_collision()  # trans vel for collision with base.
        # self.update_table_collision()

        self.update_total_vels()

        # assuming this is a
        pfield_vel = [0.0] * self.num_dof

        # update the first 6
        pfield_vel[:3] = list(self.pfieldvel_array[:])
        pfield_vel[3:6] = list(self.pfieldrot_array[:])

        return pfield_vel  # 8D list with gripper vel to be zero

    def update_attractor_vels(self):
        self.pfieldvel_array = (np.zeros((1, 3)))[0]  # might be unnecessary
        if np.linalg.norm(self.goal_position - self.eef_position) < 0.05:  # stringent condition
            self.pfieldvel_array = (np.zeros((1, 3)))[0]
        else:
            if np.linalg.norm(self.goal_position - self.eef_position) < 0.10:
                self.pfieldvel_array = self.goal_position - self.eef_position
            else:
                self.pfieldvel_array = (
                    0.1
                    * (self.goal_position - self.eef_position)
                    / (np.linalg.norm(self.goal_position - self.eef_position))
                )

    def update_attractor_rotvels(self):
        # with respect to world frame.
        # self.diff_quat = tfs.quaternion_multiply(self.goal_quat, tfs.quaternion_inverse(self.eef_quat))
        # with respect to body frame as kinova API takes care of transformation wrt world frame
        self.diff_quat = tfs.quaternion_multiply(tfs.quaternion_inverse(self.eef_quat), self.goal_quat)
        self.diff_quat = self.diff_quat / np.linalg.norm(self.diff_quat)  # normalize
        self.theta_to_goal = 2 * math.acos(self.diff_quat[3])  # 0 to 2pi. only rotation in one direction.
        if self.theta_to_goal > math.pi:  # wrap angle
            self.theta_to_goal -= 2 * math.pi
            self.theta_to_goal = abs(self.theta_to_goal)
            self.diff_quat = -self.diff_quat

        self.pfieldrot_array = (np.zeros((1, 3)))[0]
        norm_den = math.sqrt(1 - self.diff_quat[3] * self.diff_quat[3])  # norm of the denominator
        if norm_den < 0.001:
            self.pfieldrot_array[0] = self.diff_quat[0]
            self.pfieldrot_array[1] = self.diff_quat[1]
            self.pfieldrot_array[2] = self.diff_quat[2]
        else:
            self.pfieldrot_array[0] = self.diff_quat[0] / norm_den
            self.pfieldrot_array[1] = self.diff_quat[1] / norm_den
            self.pfieldrot_array[2] = self.diff_quat[2] / norm_den
            self.pfieldrot_array[:] = 0.7 * self.pfieldrot_array[:]  # sclae the velocity

        if abs(self.theta_to_goal) < 0.06:
            self.pfieldrot_array = (np.zeros((1, 3)))[0]

    def update_repeller_vels(self):
        self.repel_sum = npa([0] * 3, dtype="f")
        # print("NUM_OBS", self.num_obstacles)
        for i in range(0, self.num_obstacles):
            self.dist_to_obs[i] = np.linalg.norm(self.eef_position - self.obstacles_position[i])
            if self.dist_to_obs[i] < self.obs_threshold:
                self.iswithinobs = True
                self.repel_sum = self.repel_sum + (self.eef_position - self.obstacles_position[i]) / (
                    70 * (self.dist_to_obs[i]) ** 2 + np.finfo(np.double).tiny
                )

        if np.linalg.norm(self.repel_sum) > 0.15:
            self.repel_sum = 0.15 * self.repel_sum / np.linalg.norm(self.repel_sum)
            # print "Capped repel vel ", self.repel_sum, rospy.get_namespace()

    def update_base_collision(self):
        if np.linalg.norm(self.hand_position[:2]) < self.cylinder_rad and self.hand_position[2] < self.cylinder_h:
            stretch_matrix = np.array(
                [[0, 0], [0, 0.1]], np.float
            )  # push the velocity vector further along -y so that the robot moves away from the base more aggressively.
            stretch_vec = np.dot(stretch_matrix, self.hand_position[:2])
            # print "stretch vec",(stretch_vec)/(12*np.linalg.norm(stretch_vec))
            self.repel_sum[:2] = self.repel_sum[:2] + (stretch_vec) / (12 * np.linalg.norm(stretch_vec))
            # print "rEpel vel _base", self.repel_sum, rospy.get_namespace()
            if self.hand_position[1] > 0:
                print("Wrong side")
                self.repel_sum[1] = -math.fabs(self.repel_sum[1])  # reverse direction of y direction vel
                # print "rEpel vel", self.repel_sum
            self.iswithinbase = True

    def update_table_collision(self):
        # if the finger tips are close to the table. Just apply upward velocity. No decay or anything right now
        if self.finger_1_position[2] < 0.04 or self.finger_2_position[2] < 0.04 or self.finger_3_position[2] < 0.04:
            self.repel_sum[2] = 0.3
            self.iswithintable = True

    def update_total_vels(self):
        if not self.iswithinbase:
            self.pfieldvel_array[:] = self.pfieldvel_array[:] + self.repel_sum
        else:
            # more repelling than attractioin# same result on either if block
            self.pfieldvel_array[:] = 0.4 * self.pfieldvel_array[:] + self.repel_sum
            self.iswithinbase = False
