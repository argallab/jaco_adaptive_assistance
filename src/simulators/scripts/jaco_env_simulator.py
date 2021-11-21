#!/usr/bin/env python

# functions like the blend node
# maintains an environment class instance:
# # encapsulates where the objects and obstacles are in the world.
# # maintains robot state class?
# # can get the robot contiuous pose from the class.
# # if disamb is triggered.
# # grab the current position and convert to discrete state (encoding only position and mode)
# run disamb in 4D trans + mode space. Get the new location . Move robot to that position. Switch mode to one of the
# reset the DFT procedure.
# subscribes to interface_signal/convert to user_vel in full 6D
# subscribes to autonomy_vel
# blends velocities
# instead of calling step function, will publish blended vel on /control_input topic like the blend node.

# can update dft inference upon receiving user_vel.
# if disamb is activated. grab the current belief from DFT. Freeze it.
# Use the translation MDP for disamb. Goals are purely in translation space.


# InferenceEngine:  Does DFT. Can be triggered from the sim class after u_h has been computed via service. As opposed to directly from teleop class.
# DisambAlgo: Uses MDP description for characterizing nearby discrete states.
# MDP: MDPDescription of the 3D + Mode space - DONE
# Pfields_Node: with multi support avoidance
# Avoidance module: ? Takes current positions, goal, obstacles and computes the net autonomy vel.
# Teleop: New one, that emits interface signal - DONE
# ENV: Encapsulator - DONE

import collections
import rospy
import random
import time
from sensor_msgs.msg import Joy
from envs.jaco_robot_SE3_env import JacoRobotSE3Env
from disamb_algo.discrete_mi_disamb_algo import DiscreteMIDisambAlgo

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension, String, Int8, Float32
from teleop_nodes.msg import InterfaceSignal
from jaco_teleop.msg import CartVelCmd
from jaco_pfields_node.srv import ComputeVelocity, ComputeVelocityRequest, ComputeVelocityResponse
from jaco_pfields_node.srv import ObsDescList, ObsDescListRequest, ObsDescListResponse
from jaco_pfields_node.srv import GoalPose, GoalPoseRequest, GoalPoseResponse
from jaco_pfields_node.srv import InitPfields, InitPfieldsRequest, InitPfieldsResponse
from simulators.msg import State, DiscreteState, StringArray, ContinuousState
from simulators.srv import InitBelief, InitBeliefRequest, InitBeliefResponse
from simulators.srv import ResetBelief, ResetBeliefRequest, ResetBeliefResponse
from simulators.srv import ComputeIntent, ComputeIntentRequest, ComputeIntentResponse
from simulators.srv import QueryBelief, QueryBeliefRequest, QueryBeliefResponse
from scipy.spatial import KDTree

from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_msgs.msg import Int16

from geometry_msgs.msg import Vector3, Quaternion
from jaco_pfields_node.msg import ObsDesc
from geometry_msgs.msg import PoseStamped, Pose, Point  # for goals
from visualization_msgs.msg import Marker, MarkerArray

from mdp.mdp_discrete_3d_gridworld_with_modes import MDPDiscrete3DGridWorldWithModes
import numpy as np
import tf.transformations as tfs
import sys
import os
import copy
import numpy as np

from mdp.mdp_utils import *
from jaco_adaptive_assistance_utils import *

GRID_WIDTH = 14
GRID_DEPTH = 8
GRID_HEIGHT = 8

SPARSITY_FACTOR = 0.0
RAND_DIRECTION_FACTOR = 0.1


class Simulator(object):
    def __init__(self, subject_id, scene="2", start_mode="Y", algo_condition="disamb"):
        super(Simulator, self).__init__()
        rospy.init_node("Simulator")
        rospy.on_shutdown(self.shutdown_hook)
        self.scene = scene
        if self.scene == "1":
            self.num_objs = 4
        elif self.scene == "2":
            self.num_objs = 3
        elif self.scene == "3":
            self.num_objs = 3
        elif self.scene == "4":
            self.num_objs = 4

        self.obj_positions = np.array([[0] * 3] * self.num_objs, dtype="f")
        self.obj_quats = np.array([[0] * 4] * self.num_objs, dtype="f")

        self.called_shutdown = False
        self._initialize_publishers()

        self.mode_msg = Int16()

        self.blend_vel = CartVelCmd()
        self.robot_dim = 6
        self.finger_dim = 3
        self.num_dof = self.robot_dim + self.finger_dim
        _dim = [MultiArrayDimension()]
        self.current_p_g_given_uh = None

        _dim[0].label = "cartesian_velocity"
        _dim[0].size = 9
        _dim[0].stride = 9
        self.blend_vel.velocity.layout.dim = _dim
        self.blend_vel.velocity.data = [np.finfo(np.double).tiny] * self.num_dof  # realmin,, to avoid divide by zero

        user_vel = InterfaceSignal()
        self.input_action = {}
        self.input_action["full_control_signal"] = user_vel
        rospy.Subscriber("/user_vel", InterfaceSignal, self.joy_callback)

        # alpha from confidence function parameters
        self.confidence_threshold = 1.1 / len(self.obj_positions)
        self.confidence_max = 1.2 / len(self.obj_positions)
        self.alpha_max = 0.8
        if self.confidence_max != self.confidence_threshold:
            self.confidence_slope = float(self.alpha_max) / (self.confidence_max - self.confidence_threshold)
        else:
            self.confidence_slope = -1.0

        self.ENTROPY_THRESHOLD = 0.65
        self.AUTONOMY_TURN_TIMEOUT = 8  # in secs
        self.autonomy_turn_start_time = 0.0
        self.current_inferred_goal_id = 0
        self.subject_id = subject_id

        self.cylinder_rad = [0.2] * self.num_objs
        self.cylinder_h = 0.4

        self.obs_threshold = 0.12
        self.grid_loc = MarkerArray()
        self.goal_loc = MarkerArray()
        self.obs_loc = MarkerArray()
        self.vel_arrow = Marker()  # make a pose
        self.uh_arrow = Marker()

        self._init_obj_positions()

        self.env_params = None
        self.env_params = dict()

        rospy.loginfo("Waiting for jaco_pfields_multiple node")

        rospy.wait_for_service("/jaco_pfields_multiple/init_goal_for_pfield")
        rospy.wait_for_service("/jaco_pfields_multiple/init_obstacles_for_pfield")
        rospy.wait_for_service("/jaco_pfields_multiple/update_goal_for_pfield")
        rospy.wait_for_service("/jaco_pfields_multiple/init_pfields")
        rospy.wait_for_service("/jaco_pfields_multiple/compute_velocity")
        rospy.loginfo("jaco pfields multiple node services found! ")

        self.init_obstacles_srv = rospy.ServiceProxy("/jaco_pfields_multiple/init_obstacles_for_pfield", ObsDescList)
        self.init_obstacles_request = ObsDescListRequest()

        self.init_goal_pfield_srv = rospy.ServiceProxy("/jaco_pfields_multiple/init_goal_for_pfield", GoalPose)
        self.init_goal_pfield_request = GoalPoseRequest()

        self.update_goal_pfield_srv = rospy.ServiceProxy("/jaco_pfields_multiple/update_goal_for_pfield", GoalPose)
        self.update_goal_pfield_request = GoalPoseRequest()

        self.init_pfields_srv = rospy.ServiceProxy("/jaco_pfields_multiple/init_pfields", InitPfields)
        self.init_pfields_request = InitPfieldsRequest()

        self.compute_velocity_srv = rospy.ServiceProxy("/jaco_pfields_multiple/compute_velocity", ComputeVelocity)
        self.compute_velocity_request = ComputeVelocityRequest()

        # create world bounds
        self.world_bounds = collections.OrderedDict()

        self.world_bounds["xrange"] = collections.OrderedDict()
        self.world_bounds["yrange"] = collections.OrderedDict()
        self.world_bounds["zrange"] = collections.OrderedDict()

        self.world_bounds["xrange"]["lb"] = -0.7
        self.world_bounds["yrange"]["lb"] = -0.8
        self.world_bounds["zrange"]["lb"] = 0.0
        self.world_bounds["xrange"]["ub"] = 0.7
        self.world_bounds["yrange"]["ub"] = 0.0
        self.world_bounds["zrange"]["ub"] = 0.6

        self.env_params["world_bounds"] = self.world_bounds

        mdp_env_params = self._create_mdp_env_param_dict()

        self.env_params["all_mdp_env_params"] = mdp_env_params
        mdp_list = self._create_mdp_list(self.env_params["all_mdp_env_params"])
        self.env_params["mdp_list"] = mdp_list
        self.env_params["start_mode"] = start_mode  # or maybe from all 6D
        self.env_params["goal_positions"] = self.obj_positions
        self.env_params["goal_quats"] = self.obj_quats
        self.env_params["num_goals"] = len(self.obj_positions)
        # disamb algo specific params
        self.env_params["spatial_window_half_length"] = 3  # number of cells
        self.algo_condition = algo_condition
        self.env_params["robot_type"] = CartesianRobotType.SE3
        # kl_coeff, num_modes,
        self.env_params["kl_coeff"] = 0.6
        self.env_params["dist_coeff"] = 0.4

        self.all_Rs = [
            mdp_env_params["cell_size"]["x"],
            mdp_env_params["cell_size"]["y"],
            mdp_env_params["cell_size"]["z"],
            2 * mdp_env_params["cell_size"]["x"],
            2 * mdp_env_params["cell_size"]["y"],
            2 * mdp_env_params["cell_size"]["z"],
            np.linalg.norm([mdp_env_params["cell_size"]["x"], mdp_env_params["cell_size"]["y"]]),
            np.linalg.norm([mdp_env_params["cell_size"]["y"], mdp_env_params["cell_size"]["z"]]),
            np.linalg.norm([mdp_env_params["cell_size"]["x"], mdp_env_params["cell_size"]["z"]]),
            np.linalg.norm([2 * mdp_env_params["cell_size"]["x"], 2 * mdp_env_params["cell_size"]["y"]]),
            np.linalg.norm([2 * mdp_env_params["cell_size"]["x"], 2 * mdp_env_params["cell_size"]["z"]]),
            np.linalg.norm([2 * mdp_env_params["cell_size"]["y"], 2 * mdp_env_params["cell_size"]["z"]]),
            np.linalg.norm([2 * mdp_env_params["cell_size"]["x"], mdp_env_params["cell_size"]["y"]]),
            np.linalg.norm([2 * mdp_env_params["cell_size"]["x"], mdp_env_params["cell_size"]["z"]]),
            np.linalg.norm([2 * mdp_env_params["cell_size"]["y"], mdp_env_params["cell_size"]["x"]]),
            np.linalg.norm([2 * mdp_env_params["cell_size"]["y"], mdp_env_params["cell_size"]["z"]]),
            np.linalg.norm([2 * mdp_env_params["cell_size"]["z"], mdp_env_params["cell_size"]["x"]]),
            np.linalg.norm([2 * mdp_env_params["cell_size"]["z"], mdp_env_params["cell_size"]["y"]]),
            np.linalg.norm(
                [mdp_env_params["cell_size"]["x"], mdp_env_params["cell_size"]["y"], mdp_env_params["cell_size"]["z"]]
            ),
            np.linalg.norm(
                [
                    2 * mdp_env_params["cell_size"]["x"],
                    2 * mdp_env_params["cell_size"]["y"],
                    2 * mdp_env_params["cell_size"]["z"],
                ]
            ),
        ]

        self._init_goal_pfields()
        self._init_other_pfields(pfield_id="disamb")
        self._init_other_pfields(pfield_id="generic")

        self.env = JacoRobotSE3Env(self.env_params)
        # self.env.initialize()
        self.env.reset()

        self.disamb_algo = DiscreteMIDisambAlgo(self.env_params, subject_id)

        # map from x,y,z,.... to 1,2,3,...
        self.mode_msg.data = CARTESIAN_DIM_TO_CTRL_INDEX_MAP[CartesianRobotType.SE3][self.env_params["start_mode"]] + 1
        self.modepub.publish(self.mode_msg)
        # setup all services
        rospy.loginfo("Waiting for jaco_intent inference node")
        rospy.wait_for_service("/jaco_intent_inference/init_belief")
        rospy.wait_for_service("/jaco_intent_inference/reset_belief")
        rospy.wait_for_service("/jaco_intent_inference/query_belief")
        rospy.wait_for_service("/jaco_intent_inference/compute_intent")
        rospy.wait_for_service("/jaco_intent_inference/freeze_update")
        rospy.wait_for_service("/jaco_intent_inference/init_goal_locations")
        rospy.loginfo("jaco intent inference node service found! ")

        self.init_belief_srv = rospy.ServiceProxy("/jaco_intent_inference/init_belief", InitBelief)
        self.init_belief_request = InitBeliefRequest()
        self.init_belief_request.num_goals = len(self.obj_positions)
        status = self.init_belief_srv(self.init_belief_request)

        self.reset_belief_srv = rospy.ServiceProxy("/jaco_intent_inference/reset_belief", ResetBelief)
        self.reset_belief_request = ResetBeliefRequest()

        self.query_belief_srv = rospy.ServiceProxy("/jaco_intent_inference/query_belief", QueryBelief)
        self.query_belief_request = QueryBeliefRequest()
        self.current_p_g_given_uh = np.array(self.query_belief_srv(self.query_belief_request).current_p_g_given_uh)

        self.compute_intent_srv = rospy.ServiceProxy("/jaco_intent_inference/compute_intent", ComputeIntent)
        self.compute_intent_request = ComputeIntentRequest()

        self.init_goal_locations_srv = rospy.ServiceProxy("/jaco_intent_inference/init_goal_locations", ObsDescList)
        self.init_goal_locations_request = ObsDescListRequest()

        self.freeze_update_srv = rospy.ServiceProxy("/jaco_intent_inference/freeze_update", SetBool)
        self.freeze_update_request = SetBoolRequest()

        self._init_goal_locations_in_inference()

        r = rospy.Rate(100)
        self.is_autonomy_turn = False
        self.has_human_initiated = False
        self.autonomy_activate_ctr = 0
        self.DISAMB_ACTIVATE_THRESHOLD = 100
        is_done = False
        while not rospy.is_shutdown():
            if is_done:
                self.shutdown_hook("reached end of trial")
                break
            robot_position = self.env.get_robot_position()  # continuous [x,y,z]
            robot_orientation = self.env.get_robot_orientation()  # continuous 4d quat
            robot_full_state = self.env.get_robot_full_state()
            robot_hand_position = self.env.get_robot_hand_position()
            (
                robot_finger_position_1,
                robot_finger_position_2,
                robot_finger_position_3,
            ) = self.env.get_robot_finger_positions()
            robot_discrete_state = self.env.get_robot_current_discrete_state()  # (x,y,z,m) with m in [1,2,3,4,5,6]
            current_mode = robot_discrete_state[-1]  # [1,2,3,4,5,6]
            self.mode_msg.data = current_mode
            self.modepub.publish(self.mode_msg)
            human_vel = self.env.get_mode_conditioned_velocity(self.input_action["human"].interface_signal)  # uh
            is_mode_switch = self.input_action["human"].mode_switch

            self.human_vel_msg.data = list(human_vel)
            # prepare srv request for belief update
            self.compute_intent_request.robot_pose.position.x = robot_position[0]
            self.compute_intent_request.robot_pose.position.y = robot_position[1]
            self.compute_intent_request.robot_pose.position.z = robot_position[2]
            self.compute_intent_request.robot_pose.orientation.x = robot_orientation[0]
            self.compute_intent_request.robot_pose.orientation.y = robot_orientation[1]
            self.compute_intent_request.robot_pose.orientation.z = robot_orientation[2]
            self.compute_intent_request.robot_pose.orientation.w = robot_orientation[3]

            self.compute_intent_request.user_vel.velocity.data = human_vel  # 6d
            self.compute_intent_request.robot_discrete_state = list(robot_discrete_state)
            # string containing the current phm
            self.compute_intent_request.phm = self.input_action["human"].interface_action
            self.compute_intent_request.robot_vels = []  # list of autonomy vels for each goal
            # print('current discrete state, phm', robot_discrete_state, self.input_action['human'].interface_action)
            # grab autonomy vels for each goal. Need to pass as part of belief update request as well for blending
            self.compute_velocity_request.current_robot_position.x = robot_position[0]
            self.compute_velocity_request.current_robot_position.y = robot_position[1]
            self.compute_velocity_request.current_robot_position.z = robot_position[2]

            self.compute_velocity_request.current_robot_quat.x = robot_orientation[0]
            self.compute_velocity_request.current_robot_quat.y = robot_orientation[1]
            self.compute_velocity_request.current_robot_quat.z = robot_orientation[2]
            self.compute_velocity_request.current_robot_quat.w = robot_orientation[3]

            self.compute_velocity_request.current_robot_hand_position.x = robot_hand_position[0]
            self.compute_velocity_request.current_robot_hand_position.y = robot_hand_position[1]
            self.compute_velocity_request.current_robot_hand_position.z = robot_hand_position[2]

            self.compute_velocity_request.current_robot_finger_position_1.x = robot_finger_position_1[0]
            self.compute_velocity_request.current_robot_finger_position_1.y = robot_finger_position_1[1]
            self.compute_velocity_request.current_robot_finger_position_1.z = robot_finger_position_1[2]

            self.compute_velocity_request.current_robot_finger_position_2.x = robot_finger_position_2[0]
            self.compute_velocity_request.current_robot_finger_position_2.y = robot_finger_position_2[1]
            self.compute_velocity_request.current_robot_finger_position_2.z = robot_finger_position_2[2]

            self.compute_velocity_request.current_robot_finger_position_3.x = robot_finger_position_3[0]
            self.compute_velocity_request.current_robot_finger_position_3.y = robot_finger_position_3[1]
            self.compute_velocity_request.current_robot_finger_position_3.z = robot_finger_position_3[2]

            for i in range(len(self.obj_positions)):
                # for each goal get pfields
                self.compute_velocity_request.pfield_id = "goal_" + str(i)
                vel_response = self.compute_velocity_srv(self.compute_velocity_request)

                a_vel = CartVelCmd()
                a_vel.velocity.data = list(vel_response.velocity_final)
                self.compute_intent_request.robot_vels.append(a_vel)

            self.compute_intent_request.current_p_g_given_uh = list(self.current_p_g_given_uh)
            # update belief
            ii_response = self.compute_intent_srv(self.compute_intent_request)
            # retrieve current belief
            self.current_p_g_given_uh = np.array(self.query_belief_srv(self.query_belief_request).current_p_g_given_uh)

            if not self.is_autonomy_turn:
                # get argmax for most confident goal
                (
                    inferred_goal_id_str,
                    inferred_goal_id,
                    inferred_goal_prob,
                    normalized_h_of_p_g_given_phm,
                    argmax_goal_ids,
                    argmax_goal_ids_str,
                ) = self._get_most_confident_goal()
                # if autonomy inferred a valid goal, then set alpha accordingly
                if inferred_goal_id_str is not None and inferred_goal_prob is not None:
                    self.current_inferred_goal_id = inferred_goal_id
                    self.compute_velocity_request.pfield_id = inferred_goal_id_str
                    vel_response = self.compute_velocity_srv(self.compute_velocity_request)
                    autonomy_vel = list(vel_response.velocity_final)
                    inferred_goal_position = self.obj_positions[inferred_goal_id]
                    # dist_weight = self._dist_based_weight(inferred_goal_position, robot_position)
                    self.alpha = self._compute_alpha(inferred_goal_prob)
                    self.inferred_goal_pub.publish(inferred_goal_id_str)
                else:
                    # no confident goal, therefore keep autonomy vel to be 0.0 and alpha to be 0.0. Purely human vel
                    autonomy_vel = list([0.0] * np.array(human_vel).shape[0])  # 0.0 autonomy vel
                    self.alpha = 0.0  # no autonomy, purely human vel
                    self.inferred_goal_pub.publish("None")

                # blend velocities
                # print('ALPHA ', self.alpha)
                self.autonomy_vel_msg.data = list(autonomy_vel)
                self._blend_velocities(np.array(human_vel), np.array(autonomy_vel))
                self.blend_vel_msg.data = list(self.blend_vel.velocity.data)
                self.alpha_msg.data = self.alpha

                if np.linalg.norm(np.array(human_vel)) < 1e-5 and is_mode_switch == False:
                    if self.has_human_initiated:
                        # zero human vel and human has already issued some non-zero velocities during their turn,
                        # in which case keep track of inactivity time
                        self.autonomy_activate_ctr += 1
                        # print("ACTIVATE AUTONOM CTR", self.autonomy_activate_ctr)
                else:
                    if not self.has_human_initiated:
                        print("HUMAN INITIATED DURING THEIR TURN")
                        self.has_human_initiated = True
                        self.has_human_initiated_pub.publish("initiated")

                        # unfreeze belief update
                        self.freeze_update_request.data = False
                        self.freeze_update_srv(self.freeze_update_request)
                        self.freeze_update_pub.publish("Unfrozen")
                    # reset the activate ctr because human is providing nonzro commands
                    self.autonomy_activate_ctr = 0

                self.blendvelpub.publish(self.blend_vel)

                # TODO: determine end condition here?

                if self.algo_condition == "disamb":
                    if self.autonomy_activate_ctr > self.DISAMB_ACTIVATE_THRESHOLD:
                        if normalized_h_of_p_g_given_phm >= self.ENTROPY_THRESHOLD:
                            print("NORMALIZED ENTROPY", normalized_h_of_p_g_given_phm)
                            print("ACTIVATING DISAMB")
                            self.turn_indicator_pub.publish("autonomy-disamb")
                            self.freeze_update_request.data = True
                            self.freeze_update_srv(self.freeze_update_request)
                            self.freeze_update_pub.publish("Frozen")
                            belief_at_disamb_time = self.current_p_g_given_uh
                            self.function_timer_pub.publish("before")
                            max_disamb_state = self.disamb_algo.get_local_disamb_state(
                                belief_at_disamb_time, robot_discrete_state, robot_position, robot_orientation
                            )
                            print("CURRENT_DISCRETE STATE", robot_discrete_state, robot_position, robot_orientation)
                            print("MAX_DISAMB STATE", max_disamb_state)
                            self.disamb_discrete_state_msg.discrete_x = max_disamb_state[0]
                            self.disamb_discrete_state_msg.discrete_y = max_disamb_state[1]
                            self.disamb_discrete_state_msg.discrete_z = max_disamb_state[2]
                            self.disamb_discrete_state_msg.discrete_mode = max_disamb_state[3]

                            (max_disamb_continuous_position, _) = self._convert_discrete_state_to_continuous_position(
                                max_disamb_state, mdp_env_params["cell_size"], self.world_bounds
                            )

                            self.autonomy_turn_target_msg.robot_position.x = max_disamb_continuous_position[0]
                            self.autonomy_turn_target_msg.robot_position.y = max_disamb_continuous_position[1]
                            self.autonomy_turn_target_msg.robot_position.z = max_disamb_continuous_position[2]
                            self.autonomy_turn_target_msg.robot_quat.x = robot_orientation[0]
                            self.autonomy_turn_target_msg.robot_quat.y = robot_orientation[1]
                            self.autonomy_turn_target_msg.robot_quat.z = robot_orientation[2]
                            self.autonomy_turn_target_msg.robot_quat.w = robot_orientation[3]

                            self.disamb_discrete_state_pub.publish(self.disamb_discrete_state_msg)
                            self.autonomy_turn_target_pub.publish(self.autonomy_turn_target_msg)

                            self.function_timer_pub.publish("after")

                            self.update_goal_pfield_request.pfield_id = "disamb"
                            self.update_goal_pfield_request.goal_position.x = max_disamb_continuous_position[0]
                            self.update_goal_pfield_request.goal_position.y = max_disamb_continuous_position[1]
                            self.update_goal_pfield_request.goal_position.z = max_disamb_continuous_position[2]
                            self.update_goal_pfield_request.goal_orientation.x = robot_orientation[0]
                            self.update_goal_pfield_request.goal_orientation.y = robot_orientation[1]
                            self.update_goal_pfield_request.goal_orientation.z = robot_orientation[2]
                            self.update_goal_pfield_request.goal_orientation.w = robot_orientation[3]
                            self.update_goal_pfield_srv(self.update_goal_pfield_request)

                            self.is_autonomy_turn = True
                            self.autonomy_turn_start_time = time.time()
                            disamb_state_mode_index = max_disamb_state[-1]
                            if disamb_state_mode_index != robot_discrete_state[-1]:
                                pass
                            self.env.set_mode_in_robot(disamb_state_mode_index)
                        else:
                            # human has stopped. autonomy' turn. Upon waiting, the confidence is still high. Therefore, move the robot to current confident goal.
                            print("ACTIVATING AUTONOMY")
                            self.turn_indicator_pub.publish("autonomy-pfield")
                            self.freeze_update_request.data = True
                            self.freeze_update_srv(self.freeze_update_request)
                            self.freeze_update_pub.publish("Frozen")
                            belief_at_disamb_time = self.current_p_g_given_uh
                            inferred_goal_position = self.obj_positions[inferred_goal_id]
                            # inferred_goal_pose = self.obj_positions[inferred_goal_id]
                            self.function_timer_pub.publish("before")
                            target_point = self._get_target_along_line(robot_position, inferred_goal_position)
                            max_disamb_continuous_position = target_point

                            self.autonomy_turn_target_msg.robot_position.x = target_point[0]
                            self.autonomy_turn_target_msg.robot_position.y = target_point[1]
                            self.autonomy_turn_target_msg.robot_position.z = target_point[2]
                            self.autonomy_turn_target_msg.robot_quat.x = robot_orientation[0]
                            self.autonomy_turn_target_msg.robot_quat.y = robot_orientation[1]
                            self.autonomy_turn_target_msg.robot_quat.z = robot_orientation[2]
                            self.autonomy_turn_target_msg.robot_quat.w = robot_orientation[3]
                            self.autonomy_turn_start_time = time.time()
                            self.autonomy_turn_target_pub.publish(self.autonomy_turn_target_msg)
                            self.function_timer_pub.publish("after")

                            self.update_goal_pfield_request.pfield_id = "disamb"
                            self.update_goal_pfield_request.goal_position.x = target_point[0]
                            self.update_goal_pfield_request.goal_position.y = target_point[1]
                            self.update_goal_pfield_request.goal_position.z = target_point[2]
                            self.update_goal_pfield_request.goal_orientation.x = robot_orientation[0]
                            self.update_goal_pfield_request.goal_orientation.y = robot_orientation[1]
                            self.update_goal_pfield_request.goal_orientation.z = robot_orientation[2]
                            self.update_goal_pfield_request.goal_orientation.w = robot_orientation[3]
                            self.update_goal_pfield_srv(self.update_goal_pfield_request)
                            self.is_autonomy_turn = True

                elif self.algo_condition == "control":
                    if self.autonomy_activate_ctr > self.DISAMB_ACTIVATE_THRESHOLD:
                        print("ACTIVATING AUTONOMY")

                        self.turn_indicator_pub.publish("autonomy-control")
                        self.freeze_update_request.data = True
                        self.freeze_update_srv(self.freeze_update_request)
                        self.freeze_update_pub.publish("Frozen")

                        belief_at_disamb_time = self.current_p_g_given_uh
                        # argmax_goal_position = self.obj_positions[argmax_goal_ids]
                        self.argmax_goal_pub.publish(argmax_goal_ids_str)

                        self.function_timer_pub.publish("before")
                        destination_for_controller = self._get_mean_of_all_argmax_goals(argmax_goal_ids)
                        target_point = self._get_target_along_line(robot_position, destination_for_controller)

                        self.autonomy_turn_target_msg.robot_position.x = target_point[0]
                        self.autonomy_turn_target_msg.robot_position.y = target_point[1]
                        self.autonomy_turn_target_msg.robot_position.z = target_point[2]
                        self.autonomy_turn_target_msg.robot_quat.x = robot_orientation[0]
                        self.autonomy_turn_target_msg.robot_quat.y = robot_orientation[1]
                        self.autonomy_turn_target_msg.robot_quat.z = robot_orientation[2]
                        self.autonomy_turn_target_msg.robot_quat.w = robot_orientation[3]

                        self.autonomy_turn_start_time = time.time()
                        self.autonomy_turn_target_pub.publish(self.autonomy_turn_target_msg)
                        self.function_timer_pub.publish("after")

                        self.update_goal_pfield_request.pfield_id = "generic"
                        self.update_goal_pfield_request.goal_position.x = target_point[0]
                        self.update_goal_pfield_request.goal_position.y = target_point[1]
                        self.update_goal_pfield_request.goal_position.z = target_point[2]
                        self.update_goal_pfield_request.goal_orientation.x = robot_orientation[0]
                        self.update_goal_pfield_request.goal_orientation.y = robot_orientation[1]
                        self.update_goal_pfield_request.goal_orientation.z = robot_orientation[2]
                        self.update_goal_pfield_request.goal_orientation.w = robot_orientation[3]
                        self.update_goal_pfield_srv(self.update_goal_pfield_request)
                        self.is_autonomy_turn = True

                # end condition check
                for g_position, g_quat in zip(self.obj_positions, self.obj_quats):
                    if np.linalg.norm(g_position - robot_position) < 0.05:
                        diff_quat = tfs.quaternion_multiply(tfs.quaternion_inverse(robot_orientation), g_quat)
                        diff_quat = diff_quat / np.linalg.norm(diff_quat)  # normalize
                        theta_to_goal = 2 * math.acos(diff_quat[3])  # 0 to 2pi. only rotation in one direction.
                        if theta_to_goal > math.pi:  # wrap angle
                            theta_to_goal -= 2 * math.pi
                            theta_to_goal = abs(theta_to_goal)
                            diff_quat = -diff_quat

                        if abs(theta_to_goal) < 0.06 and self.has_human_initiated:
                            is_done = True
                            break
            else:
                # what to do during autonomy turn
                if self.algo_condition == "disamb":
                    self.compute_velocity_request.pfield_id = "disamb"
                elif self.algo_condition == "control":
                    self.compute_velocity_request.pfield_id = "generic"

                vel_response = self.compute_velocity_srv(self.compute_velocity_request)
                autonomy_vel = list(vel_response.velocity_final)
                self.autonomy_vel_msg.data = list(autonomy_vel)

                self.alpha_msg.data = 1.0  # full autonomy

                if self.algo_condition == "disamb":
                    # add condition for timeout when there is clash with pfields velocities
                    if (
                        np.linalg.norm(np.array(robot_position) - np.array(max_disamb_continuous_position)) < 0.05
                        or time.time() - self.autonomy_turn_start_time > self.AUTONOMY_TURN_TIMEOUT
                    ):
                        print("DONE WITH DISAMB PHASE")
                        # reset belief to what it was when the disamb mode was activated.
                        print("RESET BELIEF")

                        self.reset_belief_request.num_goals = len(self.obj_positions)
                        self.reset_belief_request.p_g_given_uh = list(belief_at_disamb_time)
                        self.reset_belief_srv(self.reset_belief_request)

                        self.is_autonomy_turn = False
                        self.has_human_initiated = False
                        self.turn_indicator_pub.publish("human")
                        self.autonomy_activate_ctr = 0
                        self.autonomy_turn_start_time = 0.0
                    else:
                        # print("AUTONOMY VEL", autonomy_vel)
                        self.alpha = 1.0
                        # since alpha = 1.0, this will be purely autonomy
                        self._blend_velocities(np.array(human_vel), np.array(autonomy_vel), blend_mode="disamb")
                        self.blend_vel_msg.data = list(self.blend_vel.velocity.data)
                        self.blendvelpub.publish(self.blend_vel)
                elif self.algo_condition == "control":
                    if (
                        np.linalg.norm(np.array(robot_position) - np.array(target_point)) < 0.05
                        or time.time() - self.autonomy_turn_start_time > self.AUTONOMY_TURN_TIMEOUT
                    ):
                        print("DONE WITH AUTONOMY PHASE")
                        print("RESET BELIEF")

                        self.reset_belief_request.num_goals = len(self.obj_positions)
                        self.reset_belief_request.p_g_given_uh = list(belief_at_disamb_time)
                        self.reset_belief_srv(self.reset_belief_request)

                        self.is_autonomy_turn = False
                        self.has_human_initiated = False
                        self.turn_indicator_pub.publish("human")
                        self.autonomy_activate_ctr = 0
                        self.autonomy_turn_start_time = 0.0
                    else:
                        self.alpha = 1.0
                        self._blend_velocities(np.array(human_vel), np.array(autonomy_vel), blend_mode="control")
                        self.blend_vel_msg.data = list(self.blend_vel.velocity.data)
                        self.blendvelpub.publish(self.blend_vel)

            # populate robot state for publication
            self.robot_discrete_state.discrete_x = robot_discrete_state[0]
            self.robot_discrete_state.discrete_y = robot_discrete_state[1]
            self.robot_discrete_state.discrete_z = robot_discrete_state[2]
            self.robot_discrete_state.discrete_mode = robot_discrete_state[3]

            self.robot_state.header.frame_id = "jaco_simulator"
            self.robot_state.header.stamp = rospy.Time.now()
            self.robot_state.current_robot_position.x = robot_position[0]
            self.robot_state.current_robot_position.y = robot_position[1]
            self.robot_state.current_robot_position.z = robot_position[2]

            self.robot_state.current_robot_quat.x = robot_orientation[0]
            self.robot_state.current_robot_quat.y = robot_orientation[1]
            self.robot_state.current_robot_quat.z = robot_orientation[2]
            self.robot_state.current_robot_quat.w = robot_orientation[3]

            self.robot_state.current_robot_hand_position.x = robot_hand_position[0]
            self.robot_state.current_robot_hand_position.y = robot_hand_position[1]
            self.robot_state.current_robot_hand_position.z = robot_hand_position[2]

            self.robot_state.current_robot_finger_position_1.x = robot_finger_position_1[0]
            self.robot_state.current_robot_finger_position_1.y = robot_finger_position_1[1]
            self.robot_state.current_robot_finger_position_1.z = robot_finger_position_1[2]

            self.robot_state.current_robot_finger_position_2.x = robot_finger_position_2[0]
            self.robot_state.current_robot_finger_position_2.y = robot_finger_position_2[1]
            self.robot_state.current_robot_finger_position_2.z = robot_finger_position_2[2]

            self.robot_state.current_robot_finger_position_3.x = robot_finger_position_3[0]
            self.robot_state.current_robot_finger_position_3.y = robot_finger_position_3[1]
            self.robot_state.current_robot_finger_position_3.z = robot_finger_position_3[2]

            self.robot_state.robot_discrete_state = self.robot_discrete_state

            self.robot_state_pub.publish(self.robot_state)
            self.human_vel_pub.publish(self.human_vel_msg)
            self.autonomy_vel_pub.publish(self.autonomy_vel_msg)
            self.blend_vel_pub.publish(self.blend_vel_msg)
            self.alpha_pub.publish(self.alpha_msg)

            # make markers for rvizz
            self.update_goal_array()
            self.update_obstacle_array()
            # self.update_grid_array()
            self.make_translucent_region()
            self.make_ur_pub(robot_position, autonomy_vel)
            self.make_uh_pub(robot_position, human_vel)
            # publish robot state

            # publish markers
            self.goalpub.publish(self.goal_loc)
            self.obspub.publish(self.obs_loc)
            self.gridpub.publish(self.grid_loc)
            self.obsthreshpub.publish(self.obs_thresh_loc)
            self.arrowpub.publish(self.vel_arrow)
            self.uh_arrowpub.publish(self.uh_arrow)

            r.sleep()

    def _initialize_publishers(self):
        self.shutdown_pub = rospy.Publisher("/shutdown", String, queue_size=1)
        self.blendvelpub = rospy.Publisher("/j2s7s300_driver/in/cartesian_velocity_finger", CartVelCmd, queue_size=1)
        self.modepub = rospy.Publisher("/mi/current_mode", Int16, queue_size=1)

        # study publishers
        self.robot_state_pub = rospy.Publisher("/robot_state", State, queue_size=1)
        self.human_vel_pub = rospy.Publisher("/human_vel", Float32MultiArray, queue_size=1)
        self.autonomy_vel_pub = rospy.Publisher("/autonomy_vel", Float32MultiArray, queue_size=1)
        self.blend_vel_pub = rospy.Publisher("/blend_vel", Float32MultiArray, queue_size=1)
        self.alpha_pub = rospy.Publisher("/alpha", Float32, queue_size=1)
        self.turn_indicator_pub = rospy.Publisher("/turn_indicator", String, queue_size=1)
        self.function_timer_pub = rospy.Publisher("/function_timer", String, queue_size=1)
        self.has_human_initiated_pub = rospy.Publisher("/has_human_initiated", String, queue_size=1)
        self.freeze_update_pub = rospy.Publisher("/freeze_update", String, queue_size=1)
        self.disamb_discrete_state_pub = rospy.Publisher("/disamb_discrete_state", DiscreteState, queue_size=1)
        self.autonomy_turn_target_pub = rospy.Publisher("/autonomy_turn_target", ContinuousState, queue_size=1)
        self.argmax_goal_pub = rospy.Publisher("/argmax_goal", StringArray, queue_size=1)
        self.inferred_goal_pub = rospy.Publisher("/inferred_goal", String, queue_size=1)

        # rviz publisher
        self.gridpub = rospy.Publisher("grid_pub", MarkerArray, queue_size=1)
        self.goalpub = rospy.Publisher("goal_pub", MarkerArray, queue_size=1)
        self.obspub = rospy.Publisher("obs_pub", MarkerArray, queue_size=1)
        self.obsthreshpub = rospy.Publisher("obsthresh_pub", MarkerArray, queue_size=1)
        self.arrowpub = rospy.Publisher("arrow_pub", Marker, queue_size=1)
        self.uh_arrowpub = rospy.Publisher("uh_pub", Marker, queue_size=1)

        self.robot_state = State()
        self.robot_discrete_state = DiscreteState()
        self.disamb_discrete_state_msg = DiscreteState()
        self.autonomy_turn_target_msg = ContinuousState()
        self.human_vel_msg = Float32MultiArray()
        self.autonomy_vel_msg = Float32MultiArray()
        self.blend_vel_msg = Float32MultiArray()
        self.alpha_msg = Float32()

    def _get_mean_of_all_argmax_goals(self, argmax_goal_ids):
        argmax_goal_poses = self.obj_positions[argmax_goal_ids, :]
        mean_of_all_argmax = np.mean(argmax_goal_poses, axis=0)
        return list(mean_of_all_argmax)

    def _get_target_along_line(self, start_point, end_point, R=10.0):
        R = random.choice(self.all_Rs)  # pick a distance to move comparable to the disamb conditions
        D = np.linalg.norm(np.array(end_point) - np.array(start_point))
        R = min(R, D / 2)
        target_x = start_point[0] + (R / D) * (end_point[0] - start_point[0])
        target_y = start_point[1] + (R / D) * (end_point[1] - start_point[1])
        target_z = start_point[2] + (R / D) * (end_point[2] - start_point[2])

        target = np.array([target_x, target_y, target_z])
        print("DISTANCE ", D, R, start_point, end_point, target)
        return target

    def _convert_discrete_state_to_continuous_position(self, discrete_state, cell_size, world_bounds):
        x_coord = discrete_state[0]
        y_coord = discrete_state[1]
        z_coord = discrete_state[2]
        mode = discrete_state[3] - 1

        robot_position = [
            x_coord * cell_size["x"] + cell_size["x"] / 2.0 + world_bounds["xrange"]["lb"],
            y_coord * cell_size["y"] + cell_size["y"] / 2.0 + world_bounds["yrange"]["lb"],
            z_coord * cell_size["z"] + cell_size["z"] / 2.0 + world_bounds["zrange"]["lb"],
        ]
        start_mode = MODE_INDEX_TO_DIM[mode]

        return robot_position, start_mode

    def _dist_based_weight(self, inferred_goal_position, robot_position, D=0.4, scale_factor=5):

        d = np.linalg.norm(inferred_goal_position - np.array(robot_position))
        weight_D = 0.6
        if d <= D:
            slope = -((1.0 - weight_D) / D)
            dist_weight = slope * d + 1.0
        elif d > D:
            dist_weight = weight_D * np.exp(-(d - D))
        return dist_weight

    def _compute_alpha(self, inferred_goal_prob):
        if self.confidence_slope != -1.0:
            if inferred_goal_prob <= self.confidence_threshold:
                return 0.0
            elif inferred_goal_prob > self.confidence_threshold and inferred_goal_prob <= self.confidence_max:
                return self.confidence_slope * (inferred_goal_prob - self.confidence_threshold)
            elif inferred_goal_prob > self.confidence_max and inferred_goal_prob <= 1.0:
                return self.alpha_max
        else:
            if inferred_goal_prob <= self.confidence_threshold:
                return 0.0
            else:
                return self.alpha_max

    def _blend_velocities(self, human_vel, autonomy_vel, blend_mode="regular"):
        if blend_mode == "regular":
            if np.linalg.norm(human_vel) > 1e-5:
                for i in range(self.robot_dim):
                    self.blend_vel.velocity.data[i] = self.alpha * autonomy_vel[i] + (1.0 - self.alpha) * human_vel[i]
            else:
                for i in range(self.robot_dim):
                    self.blend_vel.velocity.data[i] = 0.0

        elif blend_mode == "disamb" or blend_mode == "control":
            for i in range(self.robot_dim):
                self.blend_vel.velocity.data[i] = self.alpha * autonomy_vel[i]

        # zero gripper velocity
        self.blend_vel.velocity.data[6] = 0.0
        self.blend_vel.velocity.data[7] = 0.0
        self.blend_vel.velocity.data[8] = 0.0

    def joy_callback(self, msg):
        self.input_action["human"] = msg

    def _create_kd_tree(self, cell_size_dict):

        cell_size_x = cell_size_dict["x"]
        cell_size_y = cell_size_dict["y"]
        cell_size_z = cell_size_dict["z"]

        world_x_lb = self.world_bounds["xrange"]["lb"]
        world_y_lb = self.world_bounds["yrange"]["lb"]
        world_z_lb = self.world_bounds["zrange"]["lb"]

        data = np.zeros((GRID_WIDTH * GRID_DEPTH * GRID_HEIGHT, 3))
        self.coord_to_continuous_position_dict = collections.OrderedDict()

        for i in range(GRID_WIDTH):
            for j in range(GRID_DEPTH):
                for k in range(GRID_HEIGHT):
                    data[(i * GRID_DEPTH + j) * GRID_HEIGHT + k, 0] = i * cell_size_x + cell_size_x / 2.0 + world_x_lb
                    data[(i * GRID_DEPTH + j) * GRID_HEIGHT + k, 1] = j * cell_size_y + cell_size_y / 2.0 + world_y_lb
                    data[(i * GRID_DEPTH + j) * GRID_HEIGHT + k, 2] = k * cell_size_z + cell_size_z / 2.0 + world_z_lb
                    self.coord_to_continuous_position_dict[(i, j, k)] = tuple(
                        data[(i * GRID_DEPTH + j) * GRID_HEIGHT + k, :]
                    )

        self.continuous_position_to_loc_coord = {v: k for k, v in self.coord_to_continuous_position_dict.items()}
        # create kd tree with the cell_center_list. Use euclidean distance in 2d space for nearest neight
        self.continuous_kd_tree = KDTree(data)

    def _create_mdp_env_param_dict(self):
        mdp_env_params = collections.OrderedDict()
        mdp_env_params["rl_algo_type"] = RlAlgoType.ValueIteration
        mdp_env_params["gamma"] = 0.96
        mdp_env_params["grid_width"] = GRID_WIDTH
        mdp_env_params["grid_depth"] = GRID_DEPTH
        mdp_env_params["grid_height"] = GRID_HEIGHT

        # for MDP we are treating JACO as a 3D robot.
        mdp_env_params["robot_type"] = CartesianRobotType.R3
        mdp_env_params["mode_set_type"] = ModeSetType.OneD

        mdp_env_params["original_mdp_obstacles"] = []
        dynamics_obs_specs = []
        mdp_env_params["cell_size"] = collections.OrderedDict()
        mdp_env_params["cell_size"]["x"] = (
            self.world_bounds["xrange"]["ub"] - self.world_bounds["xrange"]["lb"]
        ) / mdp_env_params["grid_width"]
        mdp_env_params["cell_size"]["y"] = (
            self.world_bounds["yrange"]["ub"] - self.world_bounds["yrange"]["lb"]
        ) / mdp_env_params["grid_depth"]
        mdp_env_params["cell_size"]["z"] = (
            self.world_bounds["zrange"]["ub"] - self.world_bounds["zrange"]["lb"]
        ) / mdp_env_params["grid_height"]

        self._create_kd_tree(mdp_env_params["cell_size"])
        goal_list = []
        for goal_position in self.obj_positions:
            data_index = self.continuous_kd_tree.query(tuple(goal_position))[1]
            nearest_continuous_position = self.continuous_kd_tree.data[data_index, :]
            goal_discrete_position = self.continuous_position_to_loc_coord[tuple(nearest_continuous_position)]
            goal_list.append(goal_discrete_position)

        print("MDP GOAL LIST", goal_list)

        mdp_env_params["all_goals"] = goal_list
        mdp_env_params["obstacle_penalty"] = -100
        mdp_env_params["goal_reward"] = 100
        mdp_env_params["step_penalty"] = -10
        mdp_env_params["sparsity_factor"] = SPARSITY_FACTOR
        mdp_env_params["rand_direction_factor"] = RAND_DIRECTION_FACTOR
        mdp_env_params["mdp_obstacles"] = []
        mdp_env_params["dynamic_obs_specs"] = dynamics_obs_specs

        return mdp_env_params

    def _init_other_pfields(self, pfield_id):

        self.init_obstacles_request.num_obstacles = len(self.obj_positions)
        self.init_obstacles_request.obs_descs = []
        self.init_obstacles_request.pfield_id = pfield_id
        for (obj_position, obj_quat) in zip(self.obj_positions, self.obj_quats):
            obs_desc = ObsDesc()
            obs_desc.position.x = obj_position[0]
            obs_desc.position.y = obj_position[1]
            obs_desc.position.z = obj_position[2]

            obs_desc.orientation.x = obj_quat[0]
            obs_desc.orientation.y = obj_quat[1]
            obs_desc.orientation.z = obj_quat[2]
            obs_desc.orientation.w = obj_quat[3]

            self.init_obstacles_request.obs_descs.append(obs_desc)

        assert len(self.init_obstacles_request.obs_descs) == self.init_obstacles_request.num_obstacles
        # update obstacle list for pfield_id
        self.init_obstacles_srv(self.init_obstacles_request)

        # update dummy goal info for pfield_id. Will be updated after each disamb turn
        goal_position = self.obj_positions[0]
        goal_quat = self.obj_quats[0]
        self.init_goal_pfield_request.pfield_id = pfield_id
        self.init_goal_pfield_request.goal_position.x = goal_position[0]
        self.init_goal_pfield_request.goal_position.y = goal_position[1]
        self.init_goal_pfield_request.goal_position.z = goal_position[2]
        self.init_goal_pfield_request.goal_orientation.x = goal_quat[0]
        self.init_goal_pfield_request.goal_orientation.y = goal_quat[1]
        self.init_goal_pfield_request.goal_orientation.z = goal_quat[2]
        self.init_goal_pfield_request.goal_orientation.w = goal_quat[3]
        self.init_goal_pfield_srv(self.init_goal_pfield_request)

        # initialize the pfield for pfield_id
        self.init_pfields_request.pfield_id = pfield_id
        self.init_pfields_srv(self.init_pfields_request)

    def _get_most_confident_goal(self):
        p_g_given_uh_vector = self.current_p_g_given_uh + np.finfo(self.current_p_g_given_uh.dtype).tiny
        uniform_distribution = np.array([1.0 / p_g_given_uh_vector.size] * p_g_given_uh_vector.size)
        max_entropy = -np.dot(uniform_distribution, np.log2(uniform_distribution))
        normalized_h_of_p_g_given_phm = -np.dot(p_g_given_uh_vector, np.log2(p_g_given_uh_vector)) / max_entropy

        # argmax_goal_ids = np.argmax(p_g_given_uh_vector)
        # argmax_goal_ids_str = "goal_" + str(argmax_goal_ids)

        argmax_goal_ids = np.argwhere(p_g_given_uh_vector == np.amax(p_g_given_uh_vector)).flatten().tolist()
        # argmax_goal_ids =
        # argmax_goal_ids = np.argmax(p_g_given_phm_vector)
        argmax_goal_ids_str = []
        for argmax_goal_id in argmax_goal_ids:
            argmax_goal_ids_str.append("goal_" + str(argmax_goal_id))

        if normalized_h_of_p_g_given_phm <= self.ENTROPY_THRESHOLD:
            inferred_goal_id = np.argmax(p_g_given_uh_vector)
            inferred_goal_id_str = "goal_" + str(inferred_goal_id)
            inferred_goal_prob = p_g_given_uh_vector[inferred_goal_id]
            # print(normalized_h_of_p_g_given_phm, inferred_goal_id_str)
            return (
                inferred_goal_id_str,
                inferred_goal_id,
                inferred_goal_prob,
                normalized_h_of_p_g_given_phm,
                argmax_goal_ids,
                argmax_goal_ids_str,
            )
        else:
            # if entropy not greater than threshold return None as there is no confident goal
            return None, None, None, normalized_h_of_p_g_given_phm, argmax_goal_ids, argmax_goal_ids_str

    def _init_goal_pfields(self):
        num_pfields = len(self.obj_positions)

        for pfield_id in range(num_pfields):
            # one of them is a goal the others are obstacles for this pfield
            self.init_obstacles_request.num_obstacles = len(self.obj_positions) - 1
            self.init_obstacles_request.obs_descs = []
            self.init_obstacles_request.pfield_id = "goal_" + str(pfield_id)

            # create obs list for pfield_id
            for i, (obj_position, obj_quat) in enumerate(zip(self.obj_positions, self.obj_quats)):
                if i == pfield_id:
                    continue

                obs_desc = ObsDesc()
                obs_desc.position.x = obj_position[0]
                obs_desc.position.y = obj_position[1]
                obs_desc.position.z = obj_position[2]

                obs_desc.orientation.x = obj_quat[0]
                obs_desc.orientation.y = obj_quat[1]
                obs_desc.orientation.z = obj_quat[2]
                obs_desc.orientation.w = obj_quat[3]

                self.init_obstacles_request.obs_descs.append(obs_desc)

            assert len(self.init_obstacles_request.obs_descs) == self.init_obstacles_request.num_obstacles

            # update obstacle list for pfield_id
            self.init_obstacles_srv(self.init_obstacles_request)

            # update goal info for pfield_id
            goal_position = self.obj_positions[pfield_id]
            goal_quat = self.obj_quats[pfield_id]
            self.init_goal_pfield_request.pfield_id = "goal_" + str(pfield_id)
            self.init_goal_pfield_request.goal_position.x = goal_position[0]
            self.init_goal_pfield_request.goal_position.y = goal_position[1]
            self.init_goal_pfield_request.goal_position.z = goal_position[2]
            self.init_goal_pfield_request.goal_orientation.x = goal_quat[0]
            self.init_goal_pfield_request.goal_orientation.y = goal_quat[1]
            self.init_goal_pfield_request.goal_orientation.z = goal_quat[2]
            self.init_goal_pfield_request.goal_orientation.w = goal_quat[3]

            self.init_goal_pfield_srv(self.init_goal_pfield_request)

            # initialize the pfield for pfield_id
            self.init_pfields_request.pfield_id = "goal_" + str(pfield_id)
            self.init_pfields_srv(self.init_pfields_request)

    def _init_goal_locations_in_inference(self):
        self.init_goal_locations_request.num_obstacles = len(self.obj_positions)
        self.init_goal_locations_request.obs_descs = []
        self.init_goal_locations_request.pfield_id = "inference"
        for (obj_position, obj_quat) in zip(self.obj_positions, self.obj_quats):
            obs_desc = ObsDesc()
            obs_desc.position.x = obj_position[0]
            obs_desc.position.y = obj_position[1]
            obs_desc.position.z = obj_position[2]

            obs_desc.orientation.x = obj_quat[0]
            obs_desc.orientation.y = obj_quat[1]
            obs_desc.orientation.z = obj_quat[2]
            obs_desc.orientation.w = obj_quat[3]

            self.init_goal_locations_request.obs_descs.append(obs_desc)

        assert len(self.init_goal_locations_request.obs_descs) == self.init_goal_locations_request.num_obstacles
        # update goal locations for the inference engine
        self.init_goal_locations_srv(self.init_goal_locations_request)

    def _init_obj_positions(self):
        if self.scene == "1":
            self.obj_positions[0][0] = 0.414  # custom left otp
            self.obj_positions[0][1] = -0.206
            self.obj_positions[0][2] = 0.1
            self.obj_quats[0][0] = 0.706
            self.obj_quats[0][1] = -0.016
            self.obj_quats[0][2] = -0.029
            self.obj_quats[0][3] = 0.708

            self.obj_positions[1][0] = 0.306  # door knob
            self.obj_positions[1][1] = -0.550
            self.obj_positions[1][2] = 0.256
            self.obj_quats[1][0] = 0.706
            self.obj_quats[1][1] = -0.016
            self.obj_quats[1][2] = -0.029
            self.obj_quats[1][3] = 0.708

            self.obj_positions[2][0] = -0.05  # upside down
            self.obj_positions[2][1] = -0.627
            self.obj_positions[2][2] = 0.225
            self.obj_quats[2][0] = 0.706
            self.obj_quats[2][1] = -0.016
            self.obj_quats[2][2] = -0.029
            self.obj_quats[2][3] = 0.708

            self.obj_positions[3][0] = -0.370  # door knob
            self.obj_positions[3][1] = -0.55504
            self.obj_positions[3][2] = 0.3955
            self.obj_quats[3][0] = 0.706
            self.obj_quats[3][1] = -0.016
            self.obj_quats[3][2] = -0.029
            self.obj_quats[3][3] = 0.708
        elif self.scene == "2":
            self.obj_positions[0][0] = 0.414  # custom left otp
            self.obj_positions[0][1] = -0.406
            self.obj_positions[0][2] = 0.2
            self.obj_quats[0][0] = 0.706
            self.obj_quats[0][1] = -0.016
            self.obj_quats[0][2] = -0.029
            self.obj_quats[0][3] = 0.708

            self.obj_positions[1][0] = 0.0  # custom left otp
            self.obj_positions[1][1] = -0.406
            self.obj_positions[1][2] = 0.2
            self.obj_quats[1][0] = 0.706
            self.obj_quats[1][1] = -0.016
            self.obj_quats[1][2] = -0.029
            self.obj_quats[1][3] = 0.708

            self.obj_positions[2][0] = -0.514  # custom left otp
            self.obj_positions[2][1] = -0.406
            self.obj_positions[2][2] = 0.2
            self.obj_quats[2][0] = 0.706
            self.obj_quats[2][1] = -0.016
            self.obj_quats[2][2] = -0.029
            self.obj_quats[2][3] = 0.708
        elif self.scene == "3":
            self.obj_positions[0][0] = 0.1  # custom left otp
            self.obj_positions[0][1] = -0.406
            self.obj_positions[0][2] = 0.1
            self.obj_quats[0][0] = 0.706
            self.obj_quats[0][1] = -0.016
            self.obj_quats[0][2] = -0.029
            self.obj_quats[0][3] = 0.708

            self.obj_positions[1][0] = 0.1  # custom left otp
            self.obj_positions[1][1] = -0.406
            self.obj_positions[1][2] = 0.3
            self.obj_quats[1][0] = 0.706
            self.obj_quats[1][1] = -0.016
            self.obj_quats[1][2] = -0.029
            self.obj_quats[1][3] = 0.708

            self.obj_positions[2][0] = 0.1  # custom left otp
            self.obj_positions[2][1] = -0.406
            self.obj_positions[2][2] = 0.5
            self.obj_quats[2][0] = 0.706
            self.obj_quats[2][1] = -0.016
            self.obj_quats[2][2] = -0.029
            self.obj_quats[2][3] = 0.708
        elif self.scene == "4":
            # 4 object, 2 on left 2 on right
            self.obj_positions[0][0] = 0.561  # custom left otp
            self.obj_positions[0][1] = -0.262
            self.obj_positions[0][2] = 0.265
            self.obj_quats[0][0] = 0.708
            self.obj_quats[0][1] = 0.217
            self.obj_quats[0][2] = 0.339
            self.obj_quats[0][3] = 0.580

            self.obj_positions[1][0] = 0.396  # custom left otp
            self.obj_positions[1][1] = -0.490
            self.obj_positions[1][2] = 0.135
            self.obj_quats[1][0] = 0.998
            self.obj_quats[1][1] = 0.013
            self.obj_quats[1][2] = 0.058
            self.obj_quats[1][3] = -0.002

            self.obj_positions[2][0] = -0.391  # custom left otp
            self.obj_positions[2][1] = -0.574
            self.obj_positions[2][2] = 0.072
            self.obj_quats[2][0] = 0.644
            self.obj_quats[2][1] = -0.358
            self.obj_quats[2][2] = -0.206
            self.obj_quats[2][3] = 0.644

            self.obj_positions[3][0] = -0.330  # custom left otp
            self.obj_positions[3][1] = -0.570
            self.obj_positions[3][2] = 0.502
            self.obj_quats[3][0] = -0.644
            self.obj_quats[3][1] = -0.278
            self.obj_quats[3][2] = 0.673
            self.obj_quats[3][3] = -0.236

    def _create_mdp_list(self, mdp_env_params):
        mdp_list = []
        for i, g in enumerate(mdp_env_params["all_goals"]):
            mdp_env_params["mdp_goal_state"] = g
            # 3d trans goals.
            goals_that_are_obs = [
                (g_obs[0], g_obs[1], g_obs[2]) for g_obs in mdp_env_params["all_goals"] if g_obs != g
            ]
            mdp_env_params["mdp_obstacles"] = copy.deepcopy(mdp_env_params["original_mdp_obstacles"])
            mdp_env_params["mdp_obstacles"].extend(goals_that_are_obs)

            discrete_jaco_SE3_mdp = MDPDiscrete3DGridWorldWithModes(copy.deepcopy(mdp_env_params))
            mdp_list.append(discrete_jaco_SE3_mdp)

        return mdp_list

    def make_translucent_region(self):
        self.obs_thresh_loc = MarkerArray()
        obs_list = [x for x in range(0, self.num_objs) if x != self.current_inferred_goal_id]
        for i in obs_list:
            marker = Marker()
            marker.header.frame_id = "j2s7s300_link_base"
            marker.ns = "jaco_env_simulator"
            marker.id = 2 * i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = self.obj_positions[i][0]
            marker.pose.position.y = self.obj_positions[i][1]
            marker.pose.position.z = self.obj_positions[i][2]
            marker.scale.x = 2 * self.obs_threshold
            marker.scale.y = 2 * self.obs_threshold
            marker.scale.z = 2 * self.obs_threshold
            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.color.a = 0.25
            marker.color.b = 1.0
            self.obs_thresh_loc.markers.append(marker)

        self.add_base_cylinder()

    def add_base_cylinder(self):
        marker = Marker()
        marker.header.frame_id = "j2s7s300_link_base"
        marker.ns = "jaco_env_simulator"
        marker.id = 101
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0.2
        marker.scale.x = self.cylinder_rad[self.current_inferred_goal_id] * 2
        marker.scale.y = self.cylinder_rad[self.current_inferred_goal_id] * 2
        marker.scale.z = self.cylinder_h
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.color.a = 0.25
        marker.color.b = 1.0
        self.obs_thresh_loc.markers.append(marker)

    def update_goal_array(self):
        self.goal_loc = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "j2s7s300_link_base"
        marker.ns = "jaco_env_simulator"
        marker.id = 1001
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position.x = self.obj_positions[self.current_inferred_goal_id][0]
        marker.pose.position.y = self.obj_positions[self.current_inferred_goal_id][1]
        marker.pose.position.z = self.obj_positions[self.current_inferred_goal_id][2]
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.2
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.color.a = 1.0
        marker.color.b = 0.8
        self.goal_loc.markers.append(marker)

    def update_obstacle_array(self):
        self.obs_loc = MarkerArray()
        obs_list = [x for x in range(0, self.num_objs) if x != self.current_inferred_goal_id]
        # print "Obs List ", obs_list
        for i in obs_list:
            # print i
            marker = Marker()
            marker.header.frame_id = "j2s7s300_link_base"
            marker.ns = "jaco_env_simulator"
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = self.obj_positions[i][0]
            marker.pose.position.y = self.obj_positions[i][1]
            marker.pose.position.z = self.obj_positions[i][2]
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.color.a = 1.0
            marker.color.r = 1.0
            self.obs_loc.markers.append(marker)

    def update_grid_array(self):
        self.grid_loc = MarkerArray()
        i = 0
        for k, grid_loc in self.coord_to_continuous_position_dict.items():
            # v is a 3D tuple with trans locations
            marker = Marker()
            marker.header.frame_id = "j2s7s300_link_base"
            marker.ns = "jaco_env_simulator"
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = grid_loc[0]
            marker.pose.position.y = grid_loc[1]
            marker.pose.position.z = grid_loc[2]
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.color.a = 1.0
            marker.color.r = 1.0
            self.grid_loc.markers.append(marker)
            i += 1

    def make_ur_pub(self, robot_position, autonomy_vel):
        marker = Marker()
        marker.header.frame_id = "j2s7s300_link_base"
        marker.ns = "jaco_env_simulator"
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.id = 20

        p1 = Point()
        p1.x = robot_position[0]
        p1.y = robot_position[1]
        p1.z = robot_position[2]
        marker.points.append(p1)

        p2 = Point()
        if np.linalg.norm(autonomy_vel[:3]) > 0.01:
            p2.x = robot_position[0] + 0.1 * autonomy_vel[0] / np.linalg.norm(autonomy_vel[:3])
            p2.y = robot_position[1] + 0.1 * autonomy_vel[1] / np.linalg.norm(autonomy_vel[:3])
            p2.z = robot_position[2] + 0.1 * autonomy_vel[2] / np.linalg.norm(autonomy_vel[:3])
        else:
            p2.x = robot_position[0]
            p2.y = robot_position[1]
            p2.z = robot_position[2]

        marker.points.append(p2)

        marker.color.a = 1.0
        marker.color.r = 1.0

        marker.scale.x = 0.03
        marker.scale.y = 0.04
        marker.scale.z = 0.01

        self.vel_arrow = marker

    def make_uh_pub(self, robot_position, user_vel):
        marker = Marker()
        marker.header.frame_id = "j2s7s300_link_base"
        marker.ns = "jaco_env_simulator"
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.id = 20

        p1 = Point()
        p1.x = robot_position[0]
        p1.y = robot_position[1]
        p1.z = robot_position[2]
        marker.points.append(p1)

        p2 = Point()
        if np.linalg.norm(user_vel[:3]) > 0.01:
            p2.x = robot_position[0] + 0.1 * user_vel[0] / np.linalg.norm(user_vel[:3])
            p2.y = robot_position[1] + 0.1 * user_vel[1] / np.linalg.norm(user_vel[:3])
            p2.z = robot_position[2] + 0.1 * user_vel[2] / np.linalg.norm(user_vel[:3])
        else:
            p2.x = robot_position[0]
            p2.y = robot_position[1]
            p2.z = robot_position[2]

        marker.points.append(p2)

        marker.color.a = 1.0
        marker.color.b = 1.0

        marker.scale.x = 0.03
        marker.scale.y = 0.04
        marker.scale.z = 0.01

        self.uh_arrow = marker

    def shutdown_hook(self, msg_string="DONE"):
        if not self.called_shutdown:
            self.called_shutdown = True
            self.shutdown_pub.publish("shutdown")
            print("Shutting down")


if __name__ == "__main__":
    subject_id = sys.argv[1]
    scene = sys.argv[2]
    start_mode = sys.argv[3]
    algo_condition = sys.argv[4]
    Simulator(subject_id, scene, start_mode, algo_condition)
    rospy.spin()
