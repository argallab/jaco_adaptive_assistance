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
npa = np.array

class HAInput(ControlInput):

  def __init__(self):
    ControlInput.__init__(self)

    self.modeswitch_msg = ModeSwitch()
    self.mode_msg = Int16()

    # initialize variables
    self._vel_multiplier = np.ones(9)*1
    self._vel_multiplier[0:3] = 0.1
    self._vel_multiplier[4:6] = 0.2
    self.gripper_vel_limit = 2000
    self.modeSwitchCount = 0
    self._mode = 0
    self._mode_switched = False
    self._switch_direction = 1
    self._button_latch_time = 0.8
    self._old_time = rospy.get_time()
    self._lock_input = False
    self._lock_mode_switch_plate = True
    self.debounce_delay = 80
    self.last_bp = 0
    self.last_debounce_time = 0

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
      rospy.logwarn('No rosparam for max_cart_vel found...Defaulting to max linear velocity of 50 cm/s and max rotational velocity of 50 degrees/s')
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
    self.send_msg.header.frame_id = 'ha'

    self.waiting_for_release = False

    self.lock.acquire()
    try:
      self.data = self.send_msg
    finally:
      self.lock.release()

      # FUNCTIONS:
  def zero_vel(self):
    for i in range(0,9):
        self._cart_vel[i] = 0

  def set_mode(self, setmode):
    self._mode = setmode.mode_index
    print "Current mode is ", self._mode
    status = SetModeResponse()
    self.publish_mode()
    status = True
    return status

  def publish_mode(self):
    self.modepub.publish(self._mode+1)

  def publish_modeswitch(self):
    self.modeSwitchCount = self.modeSwitchCount+1
    self.modeswitch_msg.header.stamp = rospy.Time.now()
    self.modeswitch_msg.mode = self.mode_msg.data
    self.modeswitch_msg.num_switches = self.modeSwitchCount
    self.mode_switch_pub.publish(self.modeswitch_msg)
    print "Num of mode switches %d" % self.modeSwitchCount

#######################################################################################
#                           FUNCTIONS FOR SWITCHING MODES                             #
#######################################################################################
  def switchMode(self, msg):
    if msg:
      self._mode = (self._mode + 1) % 7
      print "************MODE IS NOW ", self._mode, " *******************"
      self._lock_input = True
      self.publish_modeswitch()
      self.publish_mode()

#######################################################################################
#                           FUNCTIONS FOR MOVING ROBOT ARM                            #
#######################################################################################ggg
  def move_robot_arm(self, msg):
    self.zero_vel()

    # Paradigm 1: 1-layered back plate mode switching
  # Paradigm 2: 2-layered chin switch
  # Paradigm 3: 2-layered chin switch and back plate
    ############################
    if self._paradigm == 1 or self._paradigm == 3:
      if msg:
        self._dim_switched = self.switchMode(msg.axes[1])
      self.handle_velocities(msg)

    self.handle_velocities(msg)

    self.send_msg.velocity.data = self._cart_vel
    self.send_msg.header.stamp = rospy.Time.now()


  # axes * 63 becuse joy_mouse divides by 63
  # handles inputs to determine the velocities the hand should move if in xyz main mode
  def handle_velocities(self, msg):
    if self._mode == 6:
      for i in range(6,9):
        self._cart_vel[i] = self._max_cart_vel[i] * msg.axes[0]
    else:
       self._cart_vel[self._mode] = self._max_cart_vel[self._mode] * msg.axes[0] * self._vel_multiplier[self._mode]

#######################################################################################
#                                 MAIN FUNCTIONS                                      #
#######################################################################################
# handles inputs to determine the velocities the hand should move if in gripper main mode
  def handle_threading(self):
    self.lock.acquire()
    try:
      self.data = self.send_msg
    finally:
      self.lock.release()

# the main function, determines velocities to send to robot
  def receive(self, msg):
    if msg.axes:
      # If no axes input, reset velocities to zero
      if msg.axes[0] is 0 and msg.axes[1] is 0:
        self.zero_vel()

      # Has been reset using the homing service call.
      if msg.buttons[0] is 1:
        self._mode = 0
        self.publish_mode()

       # debouncing
      elif self.last_bp != msg.axes[1]:
        self.last_debounce_time = rospy.Time.now().to_sec()
        self._lock_mode_switch_plate = False

      # prevent from constant zeroing due to debouncing
      elif self._lock_input is True:
        if not msg.axes[0] and not msg.axes[1]:
          self._lock_input = False

      elif not self._lock_mode_switch_plate and (rospy.Time.now().to_sec() - self.last_debounce_time) > self.debounce_delay:
        self.move_robot_arm(msg)
        self._lock_mode_switch_plate = True

      else:
        self.move_robot_arm(msg)

      self.last_bp = msg.axes[1]
      self.handle_threading()

  def button_cb(self, button_msg):
    if self._paradigm > 1:
      if self._paradigm == 3:
        self._switch_direction = -1
      if button_msg.data == True:
        self._mode = (self._mode + self._switch_direction) % 7
        print "************MODE IS NOW ", self._mode, " *******************"
        self._lock_input = True
        self.publish_modeswitch()
        self.publish_mode()
        self._old_time = rospy.get_time()
      elif self._paradigm == 2 and rospy.get_time() - self._old_time > self._button_latch_time:
        self._switch_direction *= -1
        print "Changed direction "


  def getDefaultData(self):
    # since this sends velocity, the default will be all zeros.
    self.lock.acquire()
    try:
      self.data = CartVelCmd()
    finally:
      self.lock.release()

  def reconfigure_cb(self, config, level):
    # rospy.loginfo("""Reconfigure Request: {Paradigm}""".format(**config))
    self._paradigm = config.headarray_paradigm
    if self._paradigm == 2:
        self._switch_direction = 1
    return config

if __name__ == '__main__':
  rospy.init_node('headarray_node', anonymous=True)
  ha = HAInput()
  ha.startSend('/user_vel')
  rospy.Subscriber('joy_cont', Joy, ha.receive)
  rospy.Subscriber('/chin_button', Bool, ha.button_cb)
  srv = Server(HeadarrayModeSwitchParadigmConfig, ha.reconfigure_cb)
  rospy.spin()
