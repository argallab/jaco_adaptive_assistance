#!/usr/bin/env python

import rospy
import collections
import numpy as np
import os
import sys

from jaco_pfields_control.pfields.jaco_pfields import JacoPfields
from jaco_pfields_node.srv import ComputeVelocity, ComputeVelocityRequest, ComputeVelocityResponse
from jaco_pfields_node.srv import ObsDescList, ObsDescListRequest, ObsDescListResponse
from jaco_pfields_node.srv import GoalPose, GoalPoseRequest, GoalPoseResponse
from jaco_pfields_node.srv import InitPfields, InitPfieldsRequest, InitPfieldsResponse

from geometry_msgs.msg import Vector3, Quaternion
from jaco_pfields_node.msg import ObsDesc


class JacoPFieldsMultipleNode(object):
    def __init__(self):
        rospy.init_node("jaco_pfields_multiple_node")
        self.jaco_pfields_dict = collections.OrderedDict()
        self.goal_position_dict = collections.OrderedDict()
        self.num_obstacles_dict = collections.OrderedDict()
        self.goal_quat_dict = collections.OrderedDict()
        self.obstacles_pose_dict = collections.OrderedDict()

        # update service types
        rospy.Service("/jaco_pfields_multiple/init_goal_for_pfield", GoalPose, self.init_goal_for_pfield)
        rospy.Service("/jaco_pfields_multiple/update_goal_for_pfield", GoalPose, self.update_goal_for_pfield)
        rospy.Service("/jaco_pfields_multiple/init_obstacles_for_pfield", ObsDescList, self.init_obstacles_for_pfield)
        rospy.Service("/jaco_pfields_multiple/init_pfields", InitPfields, self.init_pfields)
        rospy.Service("/jaco_pfields_multiple/compute_velocity", ComputeVelocity, self.compute_velocity)

    def update_goal_for_pfield(self, req):
        pfield_id = req.pfield_id

        assert pfield_id in self.jaco_pfields_dict  # need to ensure that teh JacoPfields Instance exist
        goal_position = req.goal_position
        goal_quat = req.goal_orientation
        status = self.jaco_pfields_dict[pfield_id].set_goal_position_and_quat(goal_position, goal_quat)

        response = GoalPoseResponse()
        response.success = status
        print("Initialized GOAL for ", pfield_id, self.goal_position_dict[pfield_id], self.goal_quat_dict[pfield_id])
        return response

    def init_pfields(self, req):
        pfield_id = req.pfield_id

        jaco_pfield_instance = JacoPfields()

        assert pfield_id in self.goal_position_dict
        goal_position = self.goal_position_dict[pfield_id]
        assert pfield_id in self.goal_quat_dict
        goal_quat = self.goal_quat_dict[pfield_id]
        jaco_pfield_instance.set_goal_position_and_quat(goal_position, goal_quat)

        assert pfield_id in self.obstacles_pose_dict
        obstacles_pose_list = self.obstacles_pose_dict[pfield_id]
        jaco_pfield_instance.set_obstacles_position_and_quat(obstacles_pose_list)  # list of ObsDesc

        self.jaco_pfields_dict[pfield_id] = jaco_pfield_instance  # see if copy is needed

        response = InitPfieldsResponse()
        response.success = True
        print("PFIELD INSTANCE INITIALIZED FOR ", pfield_id, self.jaco_pfields_dict[pfield_id])
        return response

    def init_goal_for_pfield(self, req):
        pfield_id = req.pfield_id
        self.goal_position_dict[pfield_id] = req.goal_position  # Vector3
        self.goal_quat_dict[pfield_id] = req.goal_orientation  # Quaternion

        response = GoalPoseResponse()
        response.success = True
        print("Initialized GOAL for ", pfield_id, self.goal_position_dict[pfield_id], self.goal_quat_dict[pfield_id])
        return response

    def init_obstacles_for_pfield(self, req):
        pfield_id = req.pfield_id
        num_obstacles = req.num_obstacles

        self.num_obstacles_dict[pfield_id] = num_obstacles
        self.obstacles_pose_dict[pfield_id] = req.obs_descs  # list of ObsDesc
        assert num_obstacles == len(self.obstacles_pose_dict[pfield_id])
        response = ObsDescListResponse()
        response.success = True
        print("Initialized obstacles for pfield id ", pfield_id, self.obstacles_pose_dict[pfield_id])
        return response

    # need something to update the goal position for disamb as well as generic pfield.
    # for both ALL goals are obstacles.

    def compute_velocity(self, req):
        current_robot_position = req.current_robot_position
        current_robot_quat = req.current_robot_quat
        current_robot_hand_position = req.current_robot_hand_position
        current_robot_finger_position_1 = req.current_robot_finger_position_1
        current_robot_finger_position_2 = req.current_robot_finger_position_2
        current_robot_finger_position_3 = req.current_robot_finger_position_3
        pfield_id = req.pfield_id

        assert pfield_id in self.jaco_pfields_dict
        pfield_vel = self.jaco_pfields_dict[pfield_id].compute_pfield_vel(
            current_robot_position,
            current_robot_quat,
            current_robot_hand_position,
            current_robot_finger_position_1,
            current_robot_finger_position_2,
            current_robot_finger_position_3,
        )

        # create response and send back the pfield_vel
        response = ComputeVelocityResponse()
        response.velocity_final = pfield_vel
        return response


if __name__ == "__main__":
    JacoPFieldsMultipleNode()
    rospy.spin()
