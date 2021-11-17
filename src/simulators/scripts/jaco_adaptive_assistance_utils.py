import math
from enum import Enum
import collections
import tf2_ros
import rospy
import numpy as np

# LISTS
# low level commands issued by the snp interface. hp = hard puff, hs= hard sip, sp = soft puff, ss = soft sip. Also the domain for ui and um
LOW_LEVEL_COMMANDS = ["Hard Puff", "Hard Sip", "Soft Puff", "Soft Sip"]

# high level actions, move_p = move in positive direction, move_n = move in negative direction, mode_r = switch mode to right, mode_l = switch mode to left. positive and negative is conditioned on mode
HIGH_LEVEL_ACTIONS = ["move_p", "move_n", "mode_r", "mode_l"]
# true mapping of a to u
TRUE_ACTION_TO_COMMAND = collections.OrderedDict(
    {
        "x": collections.OrderedDict(
            {"move_p": "Soft Puff", "move_n": "Soft Sip", "mode_r": "Hard Puff", "mode_l": "Hard Sip"}
        ),
        "y": collections.OrderedDict(
            {"move_p": "Soft Puff", "move_n": "Soft Sip", "mode_r": "Hard Puff", "mode_l": "Hard Sip"}
        ),
        "t": collections.OrderedDict(
            {"move_p": "Soft Sip", "move_n": "Soft Puff", "mode_r": "Hard Puff", "mode_l": "Hard Sip"}
        ),
        "gr": collections.OrderedDict(
            {"move_p": "Soft Puff", "move_n": "Soft Sip", "mode_r": "Hard Puff", "mode_l": "Hard Sip"}
        ),
    }
)  # for gripper mode, move_p refers to closing the gripper and move_n refers to opening the gripper
# true inverse mapping of u to a
TRUE_COMMAND_TO_ACTION = collections.OrderedDict()
for k in TRUE_ACTION_TO_COMMAND.keys():
    TRUE_COMMAND_TO_ACTION[k] = collections.OrderedDict({v: k for k, v in TRUE_ACTION_TO_COMMAND[k].items()})

# low level commands issued by the snp interface. hp = hard puff, hs= hard sip, sp = soft puff, ss = soft sip. Also the domain for ui and um
INTERFACE_LEVEL_ACTIONS = ["Hard Puff", "Hard Sip", "Soft Puff", "Soft Sip"]
# high level actions, move_p = move in positive direction, move_n = move in negative direction, mode_r = switch mode to right, mode_l = switch mode to left. positive and negative is conditioned on mode
TASK_LEVEL_ACTIONS = ["move_p", "move_n", "to_mode_r", "to_mode_l"]
# true mapping of a to phi
TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP = collections.OrderedDict(
    {"move_p": "Soft Puff", "move_n": "Soft Sip", "to_mode_r": "Hard Puff", "to_mode_l": "Hard Sip"}
)
# true inverse mapping of phi to a
TRUE_INTERFACE_ACTION_TO_TASK_ACTION_MAP = collections.OrderedDict(
    {v: k for k, v in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.items()}
)
INTERFACE_LEVEL_ACTIONS_TO_NUMBER_ID = {"Soft Puff": 0, "Soft Sip": 1, "Hard Puff": 2, "Hard Sip": 3}

# DICTIONARIES
DIM_TO_MODE_INDEX = {"x": 0, "y": 1, "z": 2, "gr": 3}
MODE_INDEX_TO_DIM = {v: k for k, v in DIM_TO_MODE_INDEX.items()}


class CartesianRobotType(Enum):
    R2 = 2  # (x,y)
    SE2_NH = 2  # (v, w)
    R3 = 3  # (x, y, z)
    SE2 = 3  # (x,y,theta)
    SE3 = 6  # (x,y,z,r,p,yaw)


class ModeSetType(Enum):
    OneD = 1
    TwoD = 2
    ThreeD = 3


class ModeTransitionType(Enum):
    Direct = 0
    Forward = 1
    Backward = 2
    Forward_Backward = 3


# note: The RobotType determines the "max" dim number that will show up in ANY of the dict values.
# THe ModeSetType will be determine the max "length" of the value string.
CARTESIAN_MODE_SET_OPTIONS = {
    CartesianRobotType.R2: {ModeSetType.OneD: {1: 1, 2: 2}, ModeSetType.TwoD: {1: 12}},
    CartesianRobotType.SE2_NH: {ModeSetType.OneD: {1: 1, 2: 2}, ModeSetType.TwoD: {1: 12}},
    CartesianRobotType.R3: {
        ModeSetType.OneD: {1: 1, 2: 2, 3: 3},
        ModeSetType.TwoD: {1: 12, 2: 23, 3: 13},
        ModeSetType.ThreeD: {1: 123},
    },
    CartesianRobotType.SE2: {
        ModeSetType.OneD: {1: 1, 2: 2, 3: 3},
        ModeSetType.TwoD: {1: 12, 2: 3},
        ModeSetType.ThreeD: {1: 123},
    },
    CartesianRobotType.SE3: {ModeSetType.OneD: {1: 1, 2: 2, 3: 3, 4: 5, 5: 4, 6: 6},},
}

CARTESIAN_DIM_NAMES = {
    CartesianRobotType.R2: ["X", "Y"],
    CartesianRobotType.SE2_NH: ["V", "W"],
    CartesianRobotType.R3: ["X", "Y", "Z"],
    CartesianRobotType.SE2: ["X", "Y", "YAW"],
    CartesianRobotType.SE3: ["X", "Y", "Z", "YAW", "PITCH", "ROLL"],
}

CARTESIAN_DIM_LABELS = {
    CartesianRobotType.R2: {"X": "L/R", "Y": "F/B"},
    CartesianRobotType.SE2_NH: {"V": "F/B", "W": "L/R"},
    CartesianRobotType.R3: {"X": "L/R", "Y": "F/B", "Z": "U/D"},
    CartesianRobotType.SE2: {"X": "L/R", "Y": "F/B", "YAW": "YAW"},
    CartesianRobotType.SE3: {"X": "L/R", "Y": "F/B", "Z": "U/D", "YAW": "YAW", "PITCH": "PITCH", "ROLL": "ROLL"},
}

CARTESIAN_DIM_TO_CTRL_INDEX_MAP = {
    CartesianRobotType.R2: {"X": 0, "Y": 1},
    CartesianRobotType.SE2_NH: {"V": 0, "W": 1},
    CartesianRobotType.R3: {"X": 0, "Y": 1, "Z": 2},
    CartesianRobotType.SE2: {"X": 0, "Y": 1, "YAW": 2},
    CartesianRobotType.SE3: {"X": 0, "Y": 1, "Z": 2, "YAW": 4, "PITCH": 3, "ROLL": 5},
}

# utility functions
def get_sign_of_number(x):
    """
    Utility function for getting the sign of a scalar. +1 for positive, -1 for negative
    """
    if int(x >= 0):
        return 1.0
    else:
        return -1.0


class JacoRobotSE3(object):
    def __init__(
        self,
        init_control_mode=1,
        robot_type=CartesianRobotType.SE3,
        mode_set_type=ModeSetType.OneD,
        mode_transition_type=ModeTransitionType.Forward_Backward,
    ):
        self.robot_type = robot_type
        self.robot_dim = self.robot_type.value  # 6
        self.mode_set_type = mode_set_type  # interface dimensionality
        self.mode_transition_type = mode_transition_type  #

        # self.mode_set = {1:1, 2:2, 3:3, 4:5, 5:4, 6:6}
        self.mode_set = CARTESIAN_MODE_SET_OPTIONS[self.robot_type][self.mode_set_type]
        self.num_modes = len(self.mode_set)  # 6, no gripper
        self.current_mode = init_control_mode  # 1,2,3 mode id. key in self.mode_set

        self.buffer = tf2_ros.Buffer(rospy.Duration(10))

        self.listener = tf2_ros.TransformListener(self.buffer)
        self.model_source_frameid = "j2s7s300_link_base"
        self.model_target_frameid = "j2s7s300_end_effector"

    # getters
    def get_position(self):
        tfpose = self.buffer.lookup_transform(
            self.model_source_frameid, self.model_target_frameid, rospy.Time(0), rospy.Duration(10)
        )
        eef_position = np.array(
            [tfpose.transform.translation.x, tfpose.transform.translation.y, tfpose.transform.translation.z], dtype="f"
        )
        return eef_position

    def get_orientation(self):
        tfpose = self.buffer.lookup_transform(
            self.model_source_frameid, self.model_target_frameid, rospy.Time(0), rospy.Duration(10)
        )
        eef_orientation = np.array(
            [
                tfpose.transform.rotation.x,
                tfpose.transform.rotation.y,
                tfpose.transform.rotation.z,
                tfpose.transform.rotation.w,
            ],
            dtype="f",
        )
        return eef_orientation

    def get_hand_position(self):
        tfpose = self.buffer.lookup_transform(
            self.model_source_frameid, "j2s7s300_link_7", rospy.Time(0), rospy.Duration(10)
        )
        hand_position = np.array(
            [tfpose.transform.translation.x, tfpose.transform.translation.y, tfpose.transform.translation.z], dtype="f"
        )
        return hand_position

    def get_finger_positions(self):
        tfpose_f1 = self.buffer.lookup_transform(
            self.model_source_frameid, "j2s7s300_link_finger_tip_1", rospy.Time(0), rospy.Duration(10)
        )
        finger_position_1 = np.array(
            [tfpose_f1.transform.translation.x, tfpose_f1.transform.translation.y, tfpose_f1.transform.translation.z],
            dtype="f",
        )
        tfpose_f2 = self.buffer.lookup_transform(
            self.model_source_frameid, "j2s7s300_link_finger_tip_2", rospy.Time(0), rospy.Duration(10)
        )
        finger_position_2 = np.array(
            [tfpose_f2.transform.translation.x, tfpose_f2.transform.translation.y, tfpose_f2.transform.translation.z],
            dtype="f",
        )
        tfpose_f3 = self.buffer.lookup_transform(
            self.model_source_frameid, "j2s7s300_link_finger_tip_3", rospy.Time(0), rospy.Duration(10)
        )
        finger_position_3 = np.array(
            [tfpose_f3.transform.translation.x, tfpose_f3.transform.translation.y, tfpose_f3.transform.translation.z],
            dtype="f",
        )
        return finger_position_1, finger_position_2, finger_position_3

    def get_state(self):
        eef_position = self.get_position()  # translation, (x,y,z) continuous
        eef_orientation = self.get_orientation()  # quaternion (x,y,z,w) continuous

        state = ((eef_position, eef_orientation), self.current_mode)  # full continuous state, in R^3 x S^3 x M
        return state

    def get_current_mode(self):
        return self.current_mode

    def set_current_mode(self, mode_index):
        self.current_mode = mode_index
        # TODO assert to mode_index type and range check

    def update_current_mode(self, mode_switch_action):
        # directly initiated from the teleop via service in env which contains this robot
        if self.mode_transition_type == ModeTransitionType.Forward_Backward:
            # ignore None mode switch action
            if mode_switch_action == "to_mode_l":
                self.current_mode = self.current_mode - 1
                if self.current_mode == 0:
                    self.current_mode = self.num_modes
                print("CURRENT_MODE", self.current_mode)
                return True
            if mode_switch_action == "to_mode_r":
                self.current_mode = self.current_mode + 1
                if self.current_mode == self.num_modes + 1:
                    self.current_mode = 1

                print("CURRENT_MODE", self.current_mode)
                return True

    def mode_conditioned_velocity(self, velocity_action):
        assert len(velocity_action) == self.mode_set_type.value  # dimensinality of the interface command
        _allowed_control_dimensions = [int(cd) - 1 for cd in str(self.mode_set[self.current_mode])]
        _all_control_dimensions = list(range(self.robot_dim))  # for velocity control
        _disallowed_control_dimensions = list(set(_all_control_dimensions) - set(_allowed_control_dimensions))
        _mappable_control_dimensions = list(range(self.mode_set_type.value))
        assert len(_mappable_control_dimensions) >= len(_allowed_control_dimensions)
        true_velocity = [0] * self.robot_dim
        for acd, mcd in zip(_allowed_control_dimensions, _mappable_control_dimensions):
            true_velocity[acd] = velocity_action[mcd]

        # flip x,y,z, so that "puff" results in rightward, forwards and downwards movement of the arm. Visually sensible.
        true_velocity[CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type]["X"]] = (
            -1.0 * true_velocity[CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type]["X"]]
        )
        true_velocity[CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type]["Y"]] = (
            -1.0 * true_velocity[CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type]["Y"]]
        )
        true_velocity[CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type]["Z"]] = (
            -1.0 * true_velocity[CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type]["Z"]]
        )

        true_velocity[CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type]["YAW"]] = (
            -1.0 * true_velocity[CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type]["YAW"]]
        )
        return true_velocity


class MicoRobotSE3(object):
    def __init__(
        self,
        init_control_mode=1,
        robot_type=CartesianRobotType.SE3,
        mode_set_type=ModeSetType.OneD,
        mode_transition_type=ModeTransitionType.Forward_Backward,
    ):
        self.robot_type = robot_type
        self.robot_dim = self.robot_type.value  # 6
        self.mode_set_type = mode_set_type  # interface dimensionality
        self.mode_transition_type = mode_transition_type  #

        # self.mode_set = {1:1, 2:2, 3:3}
        self.mode_set = CARTESIAN_MODE_SET_OPTIONS[self.robot_type][self.mode_set_type]
        self.num_modes = len(self.mode_set)
        self.current_mode = init_control_mode  # 1,2,3 mode id. key in self.mode_set

        self.buffer = tf2_ros.Buffer(rospy.Duration(10))

        self.listener = tf2_ros.TransformListener(self.buffer)
        self.model_source_frameid = "mico_link_base"
        self.model_target_frameid = "mico_end_effector"

    # getters
    def get_position(self):
        tfpose = self.buffer.lookup_transform(
            self.model_source_frameid, self.model_target_frameid, rospy.Time(0), rospy.Duration(10)
        )
        eef_position = np.array(
            [tfpose.transform.translation.x, tfpose.transform.translation.y, tfpose.transform.translation.z], dtype="f"
        )
        return eef_position

    def get_orientation(self):
        tfpose = self.buffer.lookup_transform(
            self.model_source_frameid, self.model_target_frameid, rospy.Time(0), rospy.Duration(10)
        )
        eef_orientation = np.array(
            [
                tfpose.transform.rotation.x,
                tfpose.transform.rotation.y,
                tfpose.transform.rotation.z,
                tfpose.transform.rotation.w,
            ],
            dtype="f",
        )
        return eef_orientation

    def get_hand_position(self):
        tfpose = self.buffer.lookup_transform(
            self.model_source_frameid, "mico_link_hand", rospy.Time(0), rospy.Duration(10)
        )
        hand_position = np.array(
            [tfpose.transform.translation.x, tfpose.transform.translation.y, tfpose.transform.translation.z], dtype="f"
        )
        return hand_position

    def get_finger_positions(self):
        tfpose_f1 = self.buffer.lookup_transform(
            self.model_source_frameid, "mico_link_finger_tip_1", rospy.Time(0), rospy.Duration(10)
        )
        finger_position_1 = np.array(
            [tfpose_f1.transform.translation.x, tfpose_f1.transform.translation.y, tfpose_f1.transform.translation.z],
            dtype="f",
        )
        tfpose_f2 = self.buffer.lookup_transform(
            self.model_source_frameid, "mico_link_finger_tip_2", rospy.Time(0), rospy.Duration(10)
        )
        finger_position_2 = np.array(
            [tfpose_f2.transform.translation.x, tfpose_f2.transform.translation.y, tfpose_f2.transform.translation.z],
            dtype="f",
        )
        return finger_position_1, finger_position_2

    def get_state(self):
        eef_position = self.get_position()  # translation, (x,y,z) continuous
        eef_orientation = self.get_orientation()  # quaternion (x,y,z,w) continuous

        state = ((eef_position, eef_orientation), self.current_mode)  # full continuous state, in R^3 x S^3 x M
        return state

    def get_current_mode(self):
        return self.current_mode

    def set_current_mode(self, mode_index):
        self.current_mode = mode_index
        # TODO assert to mode_index type and range check

    def update_current_mode(self, mode_switch_action):
        # directly initiated from the teleop via service in env which contains this robot
        if self.mode_transition_type == ModeTransitionType.Forward_Backward:
            # ignore None mode switch action
            if mode_switch_action == "to_mode_l":
                self.current_mode = self.current_mode - 1
                if self.current_mode == 0:
                    self.current_mode = self.num_modes
                print("CURRENT_MODE", self.current_mode)
                return True
            if mode_switch_action == "to_mode_r":
                self.current_mode = self.current_mode + 1
                if self.current_mode == self.num_modes + 1:
                    self.current_mode = 1

                print("CURRENT_MODE", self.current_mode)
                return True

    def mode_conditioned_velocity(self, velocity_action):
        assert len(velocity_action) == self.mode_set_type.value  # dimensinality of the interface command
        _allowed_control_dimensions = [int(cd) - 1 for cd in str(self.mode_set[self.current_mode])]
        _all_control_dimensions = list(range(self.robot_dim))  # for velocity control
        _disallowed_control_dimensions = list(set(_all_control_dimensions) - set(_allowed_control_dimensions))
        _mappable_control_dimensions = list(range(self.mode_set_type.value))
        assert len(_mappable_control_dimensions) >= len(_allowed_control_dimensions)
        true_velocity = [0] * self.robot_dim
        for acd, mcd in zip(_allowed_control_dimensions, _mappable_control_dimensions):
            true_velocity[acd] = velocity_action[mcd]

        return true_velocity
