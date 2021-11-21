import sys
import os
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))

from jaco_adaptive_assistance_utils import *
import numpy as np
import collections
import itertools
import rospy
import threading

from mdp.mdp_discrete_3d_gridworld_with_modes import MDPDiscrete3DGridWorldWithModes
from scipy.spatial import KDTree
import time
from envs.srv import PASAllG, PASAllGRequest, PASAllGResponse
from envs.msg import PASSingleG
from envs.srv import SwitchModeSrv, SwitchModeSrvRequest, SwitchModeSrvResponse

# maintain discrete state rep.


class JacoRobotSE3Env(object):
    def __init__(self, env_params):

        self.robot = None  # instance of JacoRobotSE3

        self.robot_position = None  # continuous position
        self.robot_orientation = None  # continuous orientation
        self.goal_poses = None  # list of goals.
        self.world_bounds = None  # workspace limits in x,y,z dimensoins?

        self.env_params = env_params

        assert self.env_params is not None
        assert "num_goals" in self.env_params
        # assert "robot_position" in self.env_params
        # assert "robot_orientation" in self.env_params

        # assert "goal_poses" in self.env_params
        assert "start_mode" in self.env_params
        assert "world_bounds" in self.env_params

        self.service_initialized = False

    def _destroy(self):
        if self.robot is None:
            return

        self.robot = None
        self.robot_position = None
        self.robot_orientation = None
        self.goal_poses = None
        self.world_bounds = None
        self.DIMENSIONS = []
        self.DIMENSION_INDICES = []

    def get_prob_a_s_all_g(self, req):
        response = PASAllGResponse()
        current_discrete_mdp_state = self._transform_continuous_robot_pose_to_discrete_state()
        # current_discrete_mdp_state = rospy.get_param("current_discrete_mdp_state", [0, 0, 0, 1])
        current_discrete_mdp_state = tuple(current_discrete_mdp_state)  # (x,y,z,m), where m is in [1,2,3]
        p_a_s_all_g = []
        optimal_action_s_g = []
        for g in range(self.num_goals):
            mdp_g = self.mdp_list[g]
            p_a_s_g_msg = PASSingleG()
            p_a_s_g_msg.goal_id = g
            for task_level_action in TASK_LEVEL_ACTIONS:
                p_a_s_g_msg.p_a_s_g.append(mdp_g.get_prob_a_given_s(current_discrete_mdp_state, task_level_action))
            p_a_s_all_g.append(p_a_s_g_msg)
            # optimal action to take in current state for goal g. Used to modify phm in inference node
            optimal_action_s_g.append(mdp_g.get_optimal_action(current_discrete_mdp_state, return_optimal=True))

        response.p_a_s_all_g = p_a_s_all_g
        response.optimal_action_s_g = optimal_action_s_g
        response.status = True
        return response

    def _transform_continuous_robot_pose_to_discrete_state(self):
        data_index = self.continuous_kd_tree.query(self.robot.get_position())[1]
        nearest_continuous_position = self.continuous_kd_tree.data[data_index, :]
        mdp_discrete_position = self.continuous_position_to_loc_coord[tuple(nearest_continuous_position)]

        # no orientation (ignoring for jaco disamb computation)

        current_mode_index = self.robot.get_current_mode()

        mdp_discrete_state = [
            mdp_discrete_position[0],
            mdp_discrete_position[1],
            mdp_discrete_position[2],
            current_mode_index,  # could be full 6D. But in disamb node the "appropriate" modes will be populated
        ]
        return mdp_discrete_state

    def _create_kd_tree_locations(self):
        grid_width = self.all_mdp_env_params["grid_width"]  # x
        grid_depth = self.all_mdp_env_params["grid_depth"]  # y
        grid_height = self.all_mdp_env_params["grid_height"]  # z

        cell_size_x = self.all_mdp_env_params["cell_size"]["x"]
        cell_size_y = self.all_mdp_env_params["cell_size"]["y"]
        cell_size_z = self.all_mdp_env_params["cell_size"]["z"]

        world_x_lb = self.world_bounds["xrange"]["lb"]
        world_y_lb = self.world_bounds["yrange"]["lb"]
        world_z_lb = self.world_bounds["zrange"]["lb"]

        data = np.zeros((grid_width * grid_depth * grid_height, 3))
        self.coord_to_continuous_position_dict = collections.OrderedDict()

        for i in range(grid_width):
            for j in range(grid_depth):
                for k in range(grid_height):
                    data[(i * grid_depth + j) * grid_height + k, 0] = i * cell_size_x + cell_size_x / 2.0 + world_x_lb
                    data[(i * grid_depth + j) * grid_height + k, 1] = j * cell_size_y + cell_size_y / 2.0 + world_y_lb
                    data[(i * grid_depth + j) * grid_height + k, 2] = k * cell_size_z + cell_size_z / 2.0 + world_z_lb
                    self.coord_to_continuous_position_dict[(i, j, k)] = tuple(
                        data[(i * grid_depth + j) * grid_height + k, :]
                    )

        self.continuous_position_to_loc_coord = {v: k for k, v in self.coord_to_continuous_position_dict.items()}
        # create kd tree with the cell_center_list. Use euclidean distance in 2d space for nearest neight
        self.continuous_kd_tree = KDTree(data)

    def reset(self):
        self._destroy()
        # For each goal there needs to be an MDP under the hood.
        self.all_mdp_env_params = self.env_params["all_mdp_env_params"]
        self.mdp_list = self.env_params["mdp_list"]  # MDP is in x,y,z,m space

        self.num_goals = self.env_params["num_goals"]
        self.robot_type = CartesianRobotType.SE3

        # continuous world boundaries
        self.world_bounds = self.env_params["world_bounds"]
        self._create_kd_tree_locations()

        # starting discrete mode #could be all 6d.
        self.start_mode = self.env_params["start_mode"]  # X,Y,Z,YAW,PITCH,ROLL
        # self.current_mode_index = DIM_TO_MODE_INDEX[self.start_mode]  # 0,1,2
        # starting_dimension = CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type][self.start_mode] + 1
        # self.current_mode_index = CARTESIAN_DIM_TO_MODE_MAP[starting_dimension]  # 1,2,3,4,5,6

        # self.current_mode_index = CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type][self.start_mode]
        starting_dimension = CARTESIAN_DIM_TO_CTRL_INDEX_MAP[self.robot_type][self.start_mode]  # dimensions
        self.current_mode_index = CARTESIAN_DIM_TO_MODE_MAP[self.robot_type][ModeSetType.OneD][starting_dimension]
        self.robot = JacoRobotSE3(init_control_mode=self.current_mode_index)

        if not self.service_initialized:
            rospy.Service("/sim_env/get_prob_a_s_all_g", PASAllG, self.get_prob_a_s_all_g)
            rospy.Service("/sim_env/switch_mode_in_robot", SwitchModeSrv, self.switch_mode_in_robot)
            self.service_initialized = True

    def set_mode_in_robot(self, mode_index):
        print("UPDATE MODE DIRECTLY")
        self.robot.set_current_mode(mode_index)

    def switch_mode_in_robot(self, req):
        print("IN SWITCH MODE SERVICE - JACO ENV")
        mode_switch_action = req.mode_switch_action
        response = SwitchModeSrvResponse()
        success = self.robot.update_current_mode(mode_switch_action)
        response.success = success
        return response

    def get_robot_current_discrete_state(self):
        # current_discrete_mdp_state = rospy.get_param("current_discrete_mdp_state", [0, 0, 0, 1])
        current_discrete_mdp_state = self._transform_continuous_robot_pose_to_discrete_state()
        return tuple(current_discrete_mdp_state)  # (x,y,z,m), where m is in 6D

    def get_robot_full_state(self):
        return self.robot.get_state()

    def get_robot_position(self):
        return self.robot.get_position()  # npa([x,y,z])

    def get_robot_orientation(self):
        return self.robot.get_orientation()  # npa([x,y,z,w])

    def get_robot_hand_position(self):
        return self.robot.get_hand_position()  # npa([x,y,z])

    def get_robot_finger_positions(self):
        return self.robot.get_finger_positions()

    def get_mode_conditioned_velocity(self, interface_signal):
        return self.robot.mode_conditioned_velocity(interface_signal)
