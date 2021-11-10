#!/usr/bin/env python

# This code converts the user_vel command to the proper array for the Kinova API

import rospy
import numpy as np
from jaco_teleop.msg import CartVelCmd
from jaco_teleop.srv import ControlSpace
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import String



class UserToControlInput():
    def __init__(self):

        # Get the robot type. This is Kinova's naming convention for their robots 
        self.robot_type = rospy.get_param('robot_type')
        self.is_joint_velocity = False

        self.control_space_pub = rospy.Publisher("control_space_topic", String, queue_size=1)

        self.set_publisher()
        self.t_last_pub = -1  # time of last event publish

    def user_vel_CB(self, msg):
        self.new_vel.velocity = msg.velocity
        self.new_vel.header.stamp = rospy.Time.now()
        # TO DO:
        # Control Input Frame ID should reflect how/where input is coming from
        self.new_vel.header.frame_id = 'full_teleop'
        self.control_input_pub.publish(self.new_vel)
        self.t_last_pub = rospy.get_time()

    def publish_vel_topic(self, event=None):
        if rospy.get_time()-self.t_last_pub>0.01: # if more than 0.01 seconds after previous input, publish new message on topic 1
            self.new_vel.velocity.data = np.zeros_like(np.zeros(self.command_length))
            self.new_vel.header.stamp = rospy.Time.now()
            self.new_vel.header.frame_id = ''
            self.control_input_pub.publish(self.new_vel)

    def set_publisher(self):
        '''Take space (string) and check whether input is joint or cartesian. Change
           the control space for teleoperation to joint or cartesian based on input.'''
           
        if rospy.has_param('is_joint'):
            self.is_joint_velocity = rospy.get_param('is_joint')

        # If joint velocity control, the array lenghth is the number of joints which needs to be specified, and publish on /joint_velocity_finger
        if self.is_joint_velocity:
            self.control_input_pub = rospy.Publisher('/'+self.robot_type+'_driver/in/joint_velocity_finger', CartVelCmd, queue_size=1, latch=True)
            # command_length = total joints (self.robot_type[3]=7) + # of fingers (self.robot_type[5]=3) - 1, we subtract 1 here to keep things consistent
            self.command_length = int(self.robot_type[3]) + int(self.robot_type[5]) - 1 
            label = 'joint_velocity'
        # otherwise, in cartesian control. publish on /cartesian_velocity_finger
        else:
            self.control_input_pub = rospy.Publisher('/'+self.robot_type+'_driver/in/cartesian_velocity_finger', CartVelCmd, queue_size=1, latch=True)
            self.command_length = 6 + int(self.robot_type[5]) #The array length is 6 (x,y,z,roll,pitch,yaw) plus the number of fingers (self.robot_type[5])
            label = 'cartesian_velocity'

        # declare message type as CartVelCmd
        self.new_vel = CartVelCmd()
        _dim = [MultiArrayDimension()]
        _dim[0].label = label
        _dim[0].size = self.command_length
        _dim[0].stride = self.command_length
        self.new_vel.velocity.layout.dim = _dim
        self.new_vel.velocity.data = np.zeros_like(np.zeros(self.command_length))
        self.new_vel.header.stamp = rospy.Time.now()
        self.new_vel.header.frame_id = 'full_teleop'

    def switch_control_space(self, req):
        '''This function is the callback to a ROS service that is used to change the control
           space used for teleoperation of the Kinova JACO robot. The control modes available 
           are joint space and cartesian space. Thus the arguments to the service must be 
           cartesian or joint.'''
        
        if req.data == 'cartesian':    
            rospy.set_param('is_joint', False)
            self.set_publisher()
            rospy.loginfo("Control space is " + str(req.data))

        elif req.data == 'joint':
            rospy.set_param('is_joint', True)
            self.set_publisher()
            rospy.loginfo("Control space is " + str(req.data))        
        else:
            rospy.logerr("uservel2cartvel: Service call failed. Argument must be 'cartesian' or 'joint'")

        self.control_space_pub.publish(req.data)
        
        return req.data

if __name__ == '__main__':

    rospy.init_node('uservel_to_cartesianvel', anonymous=True)
    control_input = UserToControlInput()
    user_vel_sub = rospy.Subscriber('/user_vel', CartVelCmd, control_input.user_vel_CB)
    rospy.Timer(rospy.Duration(1.0/100.0), control_input.publish_vel_topic) # publish data at 100 hz (required by kinova API)
    control_space_switch = rospy.Service('control_space', ControlSpace, control_input.switch_control_space)
    rospy.spin()
