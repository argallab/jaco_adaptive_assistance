import sys
import copy
import numpy as np
from math import pi

import rospy
import tf
import moveit_commander
from moveit_commander.conversions import pose_to_list

import moveit_msgs.msg
from std_msgs.msg import String
from kinova_msgs.srv import SwitchControllerMsg

class Navigation(object):
	
	def __init__(self):
		self.Ndofs = 7
		self.control_modes = ['velocity','trajectory']		#-- on the hardware, the default is velocity control
		# self.current_control_mode = 0
		self.current_control_mode = self.control_modes[0]				
		self.use_set_control_mode_service()					#-- ensures starting control mode is in 'velocity' as expected

		moveit_commander.roscpp_initialize(sys.argv)
		# rospy.init_node('waypoint_nav_node', anonymous=True)

		# Instantiate a `RobotCommander`_ object. Provides information such as the robot's
		# kinematic model and the robot's current joint states
		self.robot = moveit_commander.RobotCommander()

		# Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
		# for getting, setting, and updating the robot's internal understanding of the
		# surrounding world:
		self.scene = moveit_commander.PlanningSceneInterface()
	   
		# Instantiate a `MoveGroupCommander`_ object.  This object is an interface
		# to a planning group (group of joints).  In this tutorial the group is the primary
		# arm joints in the Panda robot, so we set the group's name to "panda_arm".
		# If you are using a different robot, change this value to the name of your robot
		# arm planning group.
		# This interface can be used to plan and execute motions:
		# group_name = "panda_arm"
		group_name = "arm"
		self.move_group = moveit_commander.MoveGroupCommander(group_name)
		# self.move_group = set_planner_id('RRTConnectkConfigDefault')

		self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
														moveit_msgs.msg.DisplayTrajectory,
														queue_size=20)

		self.planning_frame = self.move_group.get_planning_frame()
		print("INIT PLANNING FRAME: %s" % self.planning_frame)

		# We can also print the name of the end-effector link for this group:
		# self.eef_link = self.move_group.get_end_effector_link()
		# print "EEF LINK NAME: %s" % self.eef_link
		# self.move_group.set_planning_time(10)
		# print("PLANNING TIME: %d" % self.move_group.get_planning_time())
		# We can get a list of all the groups in the robot:
		self.group_names = self.robot.get_group_names()
		# print "Available Planning Groups:", self.robot.get_group_names()

		# Sometimes for debugging it is useful to print the entire state of the
		# robot:
		# print "Printing robot state"
		# print self.robot.get_current_state()
		# print ""
		
		print("--------------------")

	def switch_control_mode(self):
		"""
		Activates the switch of control modes
		"""
		if (self.current_control_mode not in self.control_modes):
			self.current_control_mode = self.control_modes[0]
		elif (self.current_control_mode is 'velocity'):
			self.current_control_mode = self.control_modes[1]
		else:
			self.current_control_mode = self.control_modes[0]
		self.use_set_control_mode_service()

	def use_set_control_mode_service(self):
		"""
		Calls the setControlModeService in the kinova api to set the control mode
		"""
		rospy.wait_for_service('/j2s7s300_driver/in/set_control_mode')
		try:
			setControlModeClient = rospy.ServiceProxy('/j2s7s300_driver/in/set_control_mode', SwitchControllerMsg)
			response = setControlModeClient(self.current_control_mode)
			return response
		except rospy.ServiceException as e:
			print("Unable to switch control modes...\n%s"%e)

	def custom_home_arm(self):
		"""
		This function homes the robot to a given home position.
		"""
		# _argallab_Home_1 = np.array([
		# 	0.057815,
		# 	2.468229,
		# 	3.166022,
		# 	1.074773,
		# 	8.029937,
		# 	4.598451,
		# 	0.141431
		# ])

		_r01_pilot_home_center = np.array([
			7.970257,
			3.144084, 
			-3.26779, 
			0.775318, 
			3.168553, 
			3.969858, 
			6.116281])

		_r01_pilot_home_rh = np.array([
			4.5288194169744, 
			3.122199780847945, 
			0.9722472765321162, 
			0.5396953539904149, 
			4.447815377678005, 
			4.598863216804574, 
			5.709847204534095]) 

		_r01_pilot_home_lh = np.array([
			12.837896931723728, 
			2.6147163256379335, 
			-8.48684245224768, 
			1.1887533515659063, 
			7.291425072686069, 
			4.749152321336579, 
			-2.2893637942173513])
		
		self._custom_home_position = _r01_pilot_home_rh
		self.switch_control_mode()
		print("Homing to custom home position........")
		self.plan_joint_goal(self._custom_home_position)
		self.switch_control_mode()
		print("Homing complete.")

	def plan_joint_goal(self, joint_goal):
		"""
		This function plans and executes a joint goal.
		TODO: move_group.get_current_joint_values() needs fixing - it returns an empty list, due to timing issues
		"""
		# print("Getting current joint values........")
		# We can get the joint values from the group and adjust some of the values:
		# joint_goal = self.move_group.get_current_joint_values()

		L = len(joint_goal)
		try:
			assert L == self.Ndofs
		except AssertionError as e:
			print("joint_goal is not the expected size\n" + str(e))
			print("expected size: " + str(self.Ndofs))
			print("joint_goal size: " + str(L))

		# The go command can be called with joint values, poses, or without any
		# parameters if you have already set the pose or joint target for the group
		self.move_group.go(joint_goal, wait=True)

		# Calling ``stop()`` ensures that there is no residual movement
		self.move_group.stop()

	def plan_waypoints(self, waypoints, orientations):
		"""
		TODO: needs to be tested
		"""
		planning_waypoints = []

		# update the planning frame to be the end-effector link
		eef_link = self.move_group.get_end_effector_link()
		# base_link = "panda_link0"
		base_link = "j2s7s300_link_base"
		print "PLANNING FRAME:", self.move_group.get_planning_frame()

		tf_listener = tf.TransformListener()
		try:
			tf_listener.waitForTransform(eef_link, base_link, rospy.Time(), rospy.Duration(5.0)) # transform between eef and robot base link
			(transform,rot) = tf_listener.lookupTransform(eef_link, base_link, rospy.Time(0))
			print "TRANSFORM [eef -> base]:", transform
		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
			print "COULDN'T GET TRANSFORM"

		wpose = self.move_group.get_current_pose().pose
		print "POSE BEFORE PLANNING:\n", self.move_group.get_current_pose().pose, "\n--------------------"

		# wpose.position.z -= 0.1 # move down (z)
		# wpose.position.y += 0.01 # then sideways (y)
		# wpose.position.x += 0.1 # then sideways (x)
		# print "POSE 1 PLANNED TO:\n", wpose, "\n--------------------"
		# planning_waypoints.append(copy.deepcopy(wpose))

		# wpose.position.z += 0.1 # move down (z)
		# wpose.position.y -= 0.01 # then sideways (y)
		# wpose.position.x -= 0.1 # then sideways (x)
		# print "POSE 2 PLANNED TO:\n", wpose, "\n--------------------"
		# planning_waypoints.append(copy.deepcopy(wpose))

		# transform all waypoints 
		waypoints -= waypoints[0]
		waypoints += np.array([wpose.position.x + transform[0], wpose.position.y + transform[1], wpose.position.z + transform[2]]) # use transform to make waypoints into robot base space
		waypoints_diff = np.zeros_like(waypoints)
		waypoints_diff[1:,:] = np.diff(waypoints, axis=0)
		# waypoints_diff = np.diff(waypoints,axis=0) # ignore the starting waypoint

		orientations_diff = np.zeros_like(orientations)
		orientations_diff[1:,:] = np.diff(orientations,axis=0)

		print "WAYPOINTS DIFFERENCES:"
		print waypoints_diff, "\n--------------------"

		# print "ORIENTATIONS (QUATERNIONS)"
		# orientations_q = np.zeros((orientations.shape[0], 4))
		# for i in range(orientations.shape[0]):
		#     orientations_q[i,:] = tf.transformations.quaternion_from_euler(orientations[i][2], orientations[i][1], orientations[i][0])
		# print orientations_q, "\n--------------------"
		# # self.move_group.set_pose_target([wpose.position.x + 0.1, wpose.position.y + 0.01, wpose.position.z - 0.1, wpose.orientation.x, wpose.orientation.y, wpose.orientation.z, wpose.orientation.w])

		# curr_x = wpose.position.x + waypoints_diff[0][0]
		# curr_y = wpose.position.y + waypoints_diff[0][1]
		# curr_z = wpose.position.z + waypoints_diff[0][2]
		# q = tf.transformations.quaternion_from_euler(orientations[0,2], orientations[0,1], orientations[0,0]) # r, p, y
		# self.move_group.set_pose_target([curr_x, curr_y, curr_z, q[0], q[1], q[2], q[3]]) # x, y, z, qx, qy, qz, qw
		# self.move_group.go(wait=True)
		# self.move_group.stop()

		# curr_x = curr_x + waypoints_diff[1][0]
		# curr_y = curr_y + waypoints_diff[1][1]
		# curr_z = curr_z + waypoints_diff[1][2]
		# self.move_group.set_position_target([curr_x, curr_y, curr_z, orientations[1,2], orientations[1,1], orientations[1,0]])
		# self.move_group.go(wait=True)
		# self.move_group.stop()

		# curr_x = curr_x + waypoints_diff[2][0]
		# curr_y = curr_y + waypoints_diff[2][1]
		# curr_z = curr_z + waypoints_diff[2][2]
		# self.move_group.set_position_target([curr_x, curr_y, curr_z, orientations[2,2], orientations[2,1], orientations[2,0]])
		# self.move_group.go(wait=True)
		# self.move_group.stop()

		# curr_x = curr_x + waypoints_diff[3][0]
		# curr_y = curr_y + waypoints_diff[3][1]
		# curr_z = curr_z + waypoints_diff[3][2]
		# self.move_group.set_position_target([curr_x, curr_y, curr_z, orientations[3,2], orientations[3,1], orientations[3,0]])#, wpose.orientation.y, wpose.orientation.z, wpose.orientation.w])
		# self.move_group.go(wait=True)
		# self.move_group.stop()

		for wp in range(waypoints_diff.shape[0]):
			wpose.position.x += waypoints_diff[wp][0]
			wpose.position.y += waypoints_diff[wp][1]
			wpose.position.z += waypoints_diff[wp][2]
			# q = tf.transformations.quaternion_from_euler(orientations[wp,2], orientations[wp,1], orientations[wp,0]) # r, p, y
			q = tf.transformations.quaternion_from_euler(orientations_diff[wp,2], orientations_diff[wp,1], orientations_diff[wp,0]) # r, p, y
			wpose.orientation.x += q[0]
			wpose.orientation.y += q[1]
			wpose.orientation.z += q[2]
			wpose.orientation.w += q[3]
			planning_waypoints.append(copy.deepcopy(wpose))

		# wpose.position.z -= 0.1 # move down (z)
		# wpose.position.y -= 0.2 # then sideways (y)
		# wpose.position.x += 0.1 # then sideways (x)
		# planning_waypoints.append(copy.deepcopy(wpose))

		# wpose.position.x = -8.7826657
		# wpose.position.y = 9.69952967  
		# wpose.position.z = 14.47093589
		# planning_waypoints.append(copy.deepcopy(wpose))

		# wpose.position.x = -12.13657759
		# wpose.position.y = 4.07385559
		# wpose.position.z = -11.47584403
		# planning_waypoints.append(copy.deepcopy(wpose))

		# wpose.position.x = -3.06880947
		# wpose.position.y = -8.25674339
		# wpose.position.z = -8.25674339
		# planning_waypoints.append(copy.deepcopy(wpose))

		# wpose.position.x = 0.73552439
		# wpose.position.y = -6.03161644
		# wpose.position.z = -1.40266222
		# planning_waypoints.append(copy.deepcopy(wpose))

		# plan and visualize
		(plan, fraction) = self.move_group.compute_cartesian_path(planning_waypoints,
																0.03, # eef (3 cm resolution)
																0.0, # jump threshold (disabled)
																avoid_collisions=True)

		# print "PLANNING WAYPOINTS:" 
		# for p in planning_waypoints:
		#     print p.position,'\n'
		print "FRACTION OF PLAN TAKEN:", fraction
		print "--------------------"
		# print "PLAN\n", plan, "\n--------------------"

		return plan, fraction # just plans and doesn't execute the planned path
		# return None, None

	def display_traj(self, plan):
		display_traj_msg = moveit_msgs.msg.DisplayTrajectory()
		display_traj_msg.trajectory_start = self.robot.get_current_state()
		display_traj_msg.trajectory.append(plan)

		# publish the visualized path
		self.display_trajectory_publisher.publish(display_traj_msg)
		rospy.sleep(5) # let rviz visualize the path

	def execute_plan(self, plan):
		self.move_group.execute(plan, wait=True)
		self.move_group.stop()

	def get_control_mode(self):
		return self.current_control_mode