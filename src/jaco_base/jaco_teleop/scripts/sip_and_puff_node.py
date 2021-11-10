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
from jaco_teleop.cfg import SipPuffModeSwitchParadigmConfig
from jaco_teleop.srv import SetMode, SetModeRequest, SetModeResponse
npa = np.array


class SNPInput(ControlInput):

  ''' data is a Float32Array message '''
  def __init__(self):

    ControlInput.__init__(self)

    # variables
    self.modeswitch_msg = ModeSwitch()
    self.mode_msg = Int16()
    self.old_msg = Joy()

    # Publishers
    self.mode_switch_pub = rospy.Publisher('mode_switches', ModeSwitch, queue_size=1)
    self.modepub = rospy.Publisher("/mi/current_mode", Int16, queue_size=1)

    # Services
    rospy.Service('/teleop_node/set_mode', SetMode, self.set_mode)

    # Initialize
    self.gripper_vel_limit = 2000
    self._mode = 0 # 0: X, 1: Y, 2: Z, 3: Roll, 4: Pictch, 5: Yaw, 6: Gripper
    self.modeSwitchCount = 0
    self._switch_direction = 1
    self._button_latch_time = 0.8
    self._old_time = rospy.get_time()
    self.modepub.publish(self.mode_msg)

    # Sip/Puff Control Thresholds
    # sip values are positive, puff values are negative in the axes
    # creating limits prevents unintended soft sips and puffs, and creates deadzone between upper and latch limit
    self._UPPER_SIP_LIMIT = 0.5
    self._LOWER_SIP_LIMIT = 0.002 # necessity of lower limit is debatable
    self._UPPER_PUFF_LIMIT = -0.5
    self._LOWER_PUFF_LIMIT = -0.002
    # latch limit governs the threshold for direction and main mode switches
    self._LATCH_LIMIT = 0.5
    self._lock_input = False
    self._ignore_input_counter = 0
    self._num_inputs_to_ignore = 10

    self._lin_vel_multiplier = 3 #2 # 10 cm/s
    self._ang_vel_multiplier = 5 #3.5 # 20 cm/s
    # Velocity multiplier
    self._vel_multiplier = np.ones(9)*1
    self._vel_multiplier[0:2] = 3
    self._vel_multiplier[3:5] = 5

    # load velocity limits to be sent to hand nodes
    self._max_cart_vel = np.ones(9)*0.3
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
    self.send_msg.header.frame_id = 'snp'

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

# Response to Set Mode Service (eg. if homing)
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


    # #######################################################################################
    #                           FUNCTIONS FOR SWITCHING MODES                             #
    #######################################################################################

  def getSwitchDirection(self, airPressure):
    if airPressure > self._LATCH_LIMIT:
      self._switch_direction = -1
    elif airPressure < self._LATCH_LIMIT:
      self._switch_direction = 1
    return 0

  # checks whether to switch mode, and changes x-->y-->z-->roll-->pitch-->yaw-->gripper
  def switchMode(self, airVelocity):
    switch_condition = abs(airVelocity) >= self._LATCH_LIMIT
    if switch_condition:
      self._mode = (self._mode + self._switch_direction) % 7
      print "************MODE IS NOW ", self._mode, " *******************"
      self._lock_input = True
      self.publish_modeswitch()
      self.publish_mode()
    return switch_condition

#######################################################################################
#                           FUNCTIONS FOR MOVING ROBOT ARM                            #
#######################################################################################
  # determines whether mode switch should happen, if not, moves robot arm based on mode
  def move_robot_arm(self, msg):

    self.zero_vel()

    # Paradigm 1: 1-layered mode switching
    ############################
    if self._paradigm ==1:
        self._mode_switched = self.switchMode(msg.axes[0])
        if self._mode_switched is False and self.withinLimits(msg.axes[0]):
          self.handle_velocities(msg)

    # Paradigm 2: Single layer scrolling both ways
    ############################
    if self._paradigm == 2:
        self.getSwitchDirection(msg.axes[0])
        self._mode_switched = self.switchMode(msg.axes[0])
        if self._mode_switched is False and self.withinLimits(msg.axes[0]):
          self.handle_velocities(msg)

    # # Paradigm 2: 2-layered mode switching
    # ############################
    if self._paradigm == 3:
        self._mode_switched = self.switchMode(msg.axes[0])
        self.handle_velocities(msg)

    # send the velocities to robot
    self.send_msg.velocity.data = self._cart_vel
    self.send_msg.header.stamp = rospy.Time.now()

  # checks whether within limits, otherwise air velocity in dead zone to too soft
  # written this way to make debugging easier if needed
  def withinLimits(self, airVelocity):
    if (self._LOWER_SIP_LIMIT < airVelocity < self._UPPER_SIP_LIMIT) or (self._UPPER_PUFF_LIMIT < airVelocity < self._LOWER_PUFF_LIMIT): # in sip limits
      #print "within limits"
      return 1
    else:
      #print "not within limits"
      return 0

  def handle_velocities(self, msg):
    if self._mode == 6:
        for i in range(6,9):
            self._cart_vel[i] = self._max_cart_vel[i] * msg.axes[0]
    else:
        self._cart_vel[self._mode] = self._max_cart_vel[self._mode] * msg.axes[0] * self._vel_multiplier[self._mode]

#######################################################################################
#                                 MAIN FUNCTIONS                                      #
#######################################################################################
  # handles threading for parallel programming used in receive function
  def handle_threading(self):
    self.lock.acquire()
    try:
      self.data = self.send_msg
    finally:
      self.lock.release()

  # the main function, determines velocities to send to robot
  def receive(self, msg):

    # prevent robot arm moving after done blowing, zero out velocities
    if msg.buttons[0] is 0 and msg.buttons[1] is 0: # the last input in each blow is 0 0 for buttons
      self._ignore_input_counter = 0 # the constraints get
      self.zero_vel()

    # If homing service was called (buttons are artifically set to 1 1)
    if msg.buttons[0] is 1 and msg.buttons[1] is 1: # reset mode and direction to x
      self._mode = 0
      self.publish_mode()
      # self.publish_modeswitch()

    # If mode was switched, wait for user to stop blowing
    if self._lock_input is True: # mode switch, waiting_for_release stops mode switching multiple times
      if not msg.buttons[0] and not msg.buttons[1]:
        self._lock_input = False

    # Ignore the leadup to powerful blow that leads to mode switch (ONLY FOR SIP-PUFF SYSTEM, otherwise delete)
    elif self._ignore_input_counter < self._num_inputs_to_ignore: # seems like thread issue if the number to ignore is too high
      self._ignore_input_counter +=1

    # If hard puff or sip (i.e. steep angle), means mode switch, so lock to wait for user to stop blowing
    elif abs((msg.axes[0] - self.old_msg.axes[0])/(msg.header.stamp - self.old_msg.header.stamp).to_sec()) >= 5:
      self._lock_input is True

    # If mode was not switched previously and buildup to switch is not ignored, then move the robot arm
    else:
      self.move_robot_arm(msg)

    self.old_msg = msg
    self.handle_threading()

  # Mode switch chin button
  def button_cb(self, button_msg):
      if self._paradigm ==3:
          if button_msg.data == True:
              self._mode = (self._mode + self._switch_direction) % 7
              print "************MODE IS NOW ", self._mode, " *******************"
              self._lock_input = True
              self.publish_modeswitch()
              self.publish_mode()
              self._old_time = rospy.get_time()
          elif rospy.get_time() - self._old_time > self._button_latch_time:
              self._switch_direction *= -1
              print "Changed direction "


  # function required from abstract in control_input
  def getDefaultData(self):
    # since this sends velocity, the default will be all zeros.
    self.lock.acquire()
    try:
      self.data = CartVelCmd()
    finally:
      self.lock.release()

  # Mode switching rosparam
  def reconfigure_cb(self, config, level):
    self._paradigm = config.snp_paradigm
    return config

if __name__ == '__main__':
  rospy.init_node('sip_puff_node', anonymous=True)
  snp = SNPInput()
  snp.startSend('/user_vel')
  rospy.Subscriber('joy_sip_puff', Joy, snp.receive)
  rospy.Subscriber('/chin_button', Bool, snp.button_cb)
  mode_paradigm_srv = Server(SipPuffModeSwitchParadigmConfig, snp.reconfigure_cb)
  rospy.spin()
