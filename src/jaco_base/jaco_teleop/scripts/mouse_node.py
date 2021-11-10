#!/usr/bin/env python

import rospy
import numpy as np
from control_input import ControlInput
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from std_msgs.msg import MultiArrayDimension
from jaco_teleop.msg import CartVelCmd
from jaco_teleop.msg import ModeSwitch
from std_msgs.msg import Int16
import time
import tf
import tf2_ros
import tf.transformations as tfs
from dynamic_reconfigure.server import Server
from jaco_teleop.cfg import HeadarrayModeSwitchParadigmConfig
from jaco_teleop.srv import SetMode, SetModeRequest, SetModeResponse
from kinova_msgs.srv import HomeArm
npa = np.array

class MouseInput(ControlInput):

  def __init__(self):
    ControlInput.__init__(self)

    self.modeswitch_msg = ModeSwitch()
    self.mode_msg = Int16()

    # [bda] THIS SHOULD READ n_axes FROM SOME CONFIG FILE
    n_axes = 3  # number of axes read from mouse (remove once read from config)
    self._is_3axis = True  # default assumes 3-axis input
    if n_axes is 2:
      self._is_3axis = False  # logic below considers only 2-axis as an alterative
    elif n_axes is not 3:
      rospy.loginfo('WARNING: Number of input axes is unexpected (%d)', n_axes)
    rospy.loginfo('Setting axes flag... _is_3axis is %d', self._is_3axis)

    # initialize variables
    self._vel_multiplier = np.ones(9)
    self._vel_multiplier[6:9] = 1000  # scale of fingers is very different
    if self._is_3axis:
      self._vel_multiplier[2] = 100  # mouse wheel is really slow input
      self._vel_multiplier[5] = 100
    print "vel multiplier ", self._vel_multiplier
    self.gripper_vel_limit = 2000
    self.modeSwitchCount = 0
    self._mode_switched = False
    self._old_time = rospy.get_time()

    if self._is_3axis:
      self._mode = "XYZ"
      self.mode_msg.data = 123 # initialize with xyz mode
    else:
      self._mode = "XY"
      self.mode_msg.data = 12  # initialize with xy mode

    # Publishers
    self.mode_switch_pub = rospy.Publisher('mode_switches', ModeSwitch, queue_size=1)
    self.modepub = rospy.Publisher("/mi/current_mode", Int16, queue_size=1)

    # Services
    rospy.Service('/teleop_node/set_mode', SetMode, self.set_mode)

    # load velocity limits
    if rospy.has_param('max_cart_vel'):
      self._max_cart_vel = np.array(rospy.get_param('max_cart_vel'))
    else:
      self._max_cart_vel = np.ones(9)
      self._max_cart_vel[6] = self.gripper_vel_limit
      self._max_cart_vel[7] = self.gripper_vel_limit
      self._max_cart_vel[8] = self.gripper_vel_limit
      rospy.logwarn('No rosparam for max_cart_vel found... setting default (50 cm/s, 50 degrees/s)')
    if not len(self._max_cart_vel) == 9:
      rospy.logerr('Length of max_cart_vel does not equal number of joints!')

    self._cart_vel = np.zeros(9)

    # Published velocity message
    self.send_msg = CartVelCmd()
    _dim = [MultiArrayDimension()]
    _dim[0].label = 'cartesian_velocity'
    _dim[0].size = 9
    _dim[0].stride = 9
    self.send_msg.velocity.layout.dim = _dim
    self.send_msg.velocity.data = np.zeros_like(self._cart_vel)
    self.send_msg.header.stamp = rospy.Time.now()
    self.send_msg.header.frame_id = 'mouse'

    self.lock.acquire()
    try:
      self.data = self.send_msg
    finally:
      self.lock.release()

  # Functions

  def zero_vel(self):
    for i in range(0,9):
      self._cart_vel[i] = 0

  def set_mode(self, setmode):
    print "Current mode to be in ", setmode.mode_index

    if self._is_3axis:
      if setmode.mode_index == 0:
        self._mode = 'XYZ'
      elif setmode.mode_index == 1:
        self._mode = 'RPW'
      elif setmode.mode_index == 2:
        self._mode = 'G'
      elif setmode.mode_index == 3:
        self._mode = 'HOME'

    else:
      if setmode.mode_index == 0:
        self._mode ='XY'
      elif setmode.mode_index == 1:
        self._mode ='XZ'
      elif setmode.mode_index == 2:
        self._mode = 'RP'
      elif setmode.mode_index == 3:
        self._mode = 'W'
      elif setmode.mode_index == 4:
        self._mode == 'G'
      elif setmode.mode_index == 5:
        self._mode = 'HOME'

    print "Current mode is ", self._mode
    status = SetModeResponse()
    self.publish_mode()
    status = True
    return status

  def publish_mode(self):

    # [bda] TODO: combine these (just search string)
    # [bda] TODO: add HOME mode?

    if self._is_3axis:
      if self._mode == 'XYZ':
        self.mode_msg.data = 123
      elif self._mode == 'RPW':
        self.mode_msg.data = 456
      elif self._mode == 'G':
        self.mode_msg.data = 7

    else:
      if self._mode == 'XY':
        self.mode_msg.data = 12
      elif self._mode == 'XZ':
        self.mode_msg.data = 13
      elif self._mode == 'RP':
        self.mode_msg.data = 45
      elif self._mode == 'W':
        self.mode_msg.data = 6
      elif self._mode == 'G':
        self.mode_msg.data = 7

    self.modepub.publish(self.mode_msg)
    print "Publishing mode: %s" % self._mode


  def publish_modeswitch(self):
    self.modeSwitchCount = self.modeSwitchCount+1
    self.modeswitch_msg.header.stamp = rospy.Time.now()
    self.modeswitch_msg.mode = self.mode_msg.data
    self.modeswitch_msg.num_switches = self.modeSwitchCount
    self.mode_switch_pub.publish(self.modeswitch_msg)
    print "Num of mode switches %d" % self.modeSwitchCount


#######################################################################################
#                           FUNCTIONS FOR MOVING ROBOT ARM                            #
#######################################################################################ggg
  def move_robot_arm(self, msg):

    # zero out the velocities
    self.zero_vel()

    # handle xyz velocities
    if "X" in self._mode:
      self._cart_vel[0] = -1*msg.axes[0]*self._vel_multiplier[0]  # so left-right matches mouse
    if "Y" in self._mode:
      self._cart_vel[1] = msg.axes[1]*self._vel_multiplier[1]
    if "Z" in self._mode:
      if self._is_3axis:
        self._cart_vel[2] = msg.axes[2]*self._vel_multiplier[2]   # 3rd axes of XYZ
      else:
        self._cart_vel[2] = msg.axes[1]*self._vel_multiplier[2]   # 2nd axes of XZ

    # handle wrist velocities
    if "R" in self._mode:
      self._cart_vel[3] = msg.axes[0]*self._vel_multiplier[3]
    if "P" in self._mode:
      self._cart_vel[4] = msg.axes[1]*self._vel_multiplier[4]
    if "W" in self._mode:
      if self._is_3axis:
        self._cart_vel[5] = msg.axes[2]*self._vel_multiplier[5]   # 3rd axes of RPY(W)
      else:
        self._cart_vel[5] = msg.axes[0]*self._vel_multiplier[5]   # 1st axes of Y(W)

    # handle gripper velocities
    if self._mode is "G":
      if abs(msg.axes[1])<abs(msg.axes[0]):    # x is dominant motion
        # +/- in dimension x opens/closes all 3 fingers simultaneously
        self._cart_vel[6] =  msg.axes[0]*self._vel_multiplier[6]
        self._cart_vel[7] =  msg.axes[0]*self._vel_multiplier[7]
        self._cart_vel[8] =  msg.axes[0]*self._vel_multiplier[8]
      elif abs(msg.axes[0])<abs(msg.axes[1]):  # y is dominant motion
        # +/- in dimension y opens/closes 2 fingers simultaneously
        self._cart_vel[6] =  msg.axes[1]*self._vel_multiplier[6]
        self._cart_vel[7] =  msg.axes[1]*self._vel_multiplier[7]
        self._cart_vel[8] =  0

      # # alternate paradigm that has each axis control one finger (for 3 axis input)
      # self._cart_vel[6] =  msg.axes[0]*self._vel_multiplier[6]
      # self._cart_vel[7] =  msg.axes[1]*self._vel_multiplier[7]
      # self._cart_vel[8] =  msg.axes[2]*self._vel_multiplier[8]

    self.send_msg.velocity.data = self._cart_vel
    self.send_msg.header.stamp = rospy.Time.now()
    # note will send zero velocities msg if in HOME mode (regardless of command)


  def switch_mode(self, msg):

    publish = True;

    # each button maps to one mode
    if self._is_3axis:

      if msg.buttons[0]:
        if msg.buttons[2]:
          self._mode = 'HOME'
        else:
          self._mode = 'XYZ'
      elif msg.buttons[1]:
        self._mode = 'RPW'
      elif msg.buttons[2]:
        self._mode = 'G'
      else:
        publish = False;
#        rospy.loginfo("WARNING: Nothing (expected) is populated in button press")

    # buttons cycle through modes
    else:

      # right button cycles forwards
      if msg.buttons[2]:
        if self._mode is 'XY':
          self._mode = 'XZ'
        elif self._mode is 'XZ':
          self._mode = 'RP'
        elif self._mode is 'RP':
          self._mode = 'W'
        elif self._mode is 'W':
          self._mode = 'G'
        elif self._mode is 'G':
          self._mode = 'HOME'
        elif self._mode is 'HOME':
          self._mode = 'XY'
        else:
          publish = False;
          rospy.loginfo("WARNING: Unexpected mode %s", self._mode)

      # left button cycles backwards
      elif msg.buttons[0]:
        if self._mode is 'XY':
          self._mode = 'HOME'
        elif self._mode is 'HOME':
          self._mode = 'G'
        elif self._mode is 'G':
          self._mode = 'W'
        elif self._mode is 'W':
          self._mode = 'RP'
        elif self._mode is 'RP':
          self._mode = 'XZ'
        elif self._mode is 'XZ':
          self._mode = 'XY'
        else:
          publish = False;
          rospy.loginfo("WARNING: Unexpected mode %s", self._mode)

      else:
        publish = False;
#        rospy.loginfo("WARNING: Nothing (expected) is populated in button press")

    if publish:
      self.publish_mode()
      self.publish_modeswitch()


  def home_robot(self):
    print "homing robot"
#    service_address = '/j2n6s300_driver/in/home_arm'
    service_address = '/' + self.robot_type + '_driver/in/home_arm'
    print "service_address ", service_address
    rospy.wait_for_service(service_address)
    try:
      home = rospy.ServiceProxy(service_address, HomeArm)
      home()
#      return None
    except rospy.ServiceException, e:
      print "Service call failed: %s"%e

#######################################################################################
#                                 MAIN FUNCTIONS                                      #
#######################################################################################

  # makes sure node waits for the current message to be fully published
  def handle_threading(self):
    self.lock.acquire()
    try:
      self.data = self.send_msg
    finally:
      self.lock.release()


  # the main function, determines velocities to send to robot
  def receive(self, msg):

    # handle continuous control commands (3 axes)
    if msg.axes:

      # if no axes input, reset velocities to zero
      if msg.axes[0] is 0 and msg.axes[1] is 0:
        if self._is_3axis:
          if msg.axes[2] is 0:
            self.zero_vel()
        else:
          self.zero_vel()

      self.move_robot_arm(msg)
      self.handle_threading()

    # handle mode switching (button presses)
    if msg.buttons:

      self.switch_mode(msg)

    # home the robot
    if self._mode is "HOME":
      if msg.buttons[0] and msg.buttons[2]:
        # trigger home position if confirm with 2-finger click
        self.home_robot()


  def getDefaultData(self):
    # since this sends velocity, the default will be all zeros.
    self.lock.acquire()
    try:
      self.data = CartVelCmd()
    finally:
      self.lock.release()

if __name__ == '__main__':
  rospy.init_node('mouse_node', anonymous=True)
  mouse = MouseInput()
  mouse.startSend('/user_vel')
  rospy.Subscriber('joy_cont', Joy, mouse.receive)
  rospy.spin()
