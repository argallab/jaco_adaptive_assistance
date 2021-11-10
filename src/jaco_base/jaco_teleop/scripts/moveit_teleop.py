#!/usr/bin/env python

import sys
import rospy
import moveit_msgs.msg 
import moveit_commander
from geometry_msgs.msg import PoseStamped


class MoveGroupTeleopObject(object):
	def __init__(self): 
		## First initialize `moveit_commander`_ and a `rospy`_ node:
		moveit_commander.roscpp_initialize(sys.argv)
		rospy.init_node('moveit_teleop', anonymous=True)

		## Instantiate a `RobotCommander`_ object. This object is the outer-level interface to
		## the robot:
		robot = moveit_commander.RobotCommander()

		## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
		## to one group of joints.  
		group_name = "arm"
		self.group = moveit_commander.MoveGroupCommander(group_name)

		## We create a `DisplayTrajectory`_ publisher which is used later to publish
		## trajectories for RViz to visualize:
		self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)


		self.pose_sub = rospy.Subscriber("/joy_pose", PoseStamped, self.receive, queue_size=1)
		rospy.spin()

	def receive(self, msg): 
		# Plan to desired goal pose
		self.group.set_pose_target(msg)
		plan = self.group.go(wait=True)
		# Calling `stop()` ensures that there is no residual movement
		self.group.stop()
		# It is always good to clear your targets after planning with poses.
		# Note: there is no equivalent function for clear_joint_value_targets()
		self.group.clear_pose_targets()

def main(): 
	try:
		move_group_teleop_object = MoveGroupTeleopObject()
	except rospy.ROSInterruptException:
		return
	except KeyboardInterrupt:
		return
		
if __name__ == '__main__': 
	main()