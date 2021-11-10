#!/usr/bin/env python

"""
Created to host a list of helper functions related to jaco_interaction
[Spring 2021] Note: this needs work
"""

import sys
import rospy

import moveit_msgs.msg
import moveit_commander

def get_current_state():
	group_name = "arm"

	moveit_commander.roscpp_initialize(sys.argv)
	# rospy.init_node("get_current_state", anonymous=False)
	robot = moveit_commander.RobotCommander()
	move_group = moveit_commander.MoveGroupCommander(group_name)

	pose_current = move_group.get_current_pose().pose
	print("CURRENT POSE:\n" + str(pose_current))

	return pose_current