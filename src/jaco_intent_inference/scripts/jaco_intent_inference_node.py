#!/usr/bin/python

import rospy
import numpy as np
import math
import sys
import threading
import tf.transformations as tfs
import collections
import os
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
from simulators.srv import InitBelief, InitBeliefRequest, InitBeliefResponse
from simulators.srv import ResetBelief, ResetBeliefRequest, ResetBeliefResponse
from simulators.srv import ComputeIntent, ComputeIntentRequest, ComputeIntentResponse
from simulators.srv import QueryBelief, QueryBeliefRequest, QueryBeliefResponse
from jaco_pfields_node.srv import ObsDescList, ObsDescListRequest, ObsDescListResponse

from jaco_intent_inference.msg import BeliefInfo
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from envs.srv import PASAllG, PASAllGRequest, PASAllGResponse
from jaco_adaptive_assistance_utils import *

npa = np.array


class JacoIntentInference(object):
    def __init__(self):
        rospy.init_node("jaco_intent_inference")
        self.lock = threading.Lock()

        rospy.Service("/jaco_intent_inference/init_belief", InitBelief, self.init_P_G_GIVEN_UH)
        rospy.Service("/jaco_intent_inference/reset_belief", ResetBelief, self.reset_P_G_GIVEN_UH)
        rospy.Service("/jaco_intent_inference/freeze_update", SetBool, self.freeze_update)
        rospy.Service("/jaco_intent_inference/init_goal_locations", ObsDescList, self.init_goal_locations)
        rospy.Service("/jaco_intent_inference/compute_intent", ComputeIntent, self.compute_belief_update)
        rospy.Service("/jaco_intent_inference/query_belief", QueryBelief, self.query_belief)

        self.belief_info_pub = rospy.Publisher("/belief_info", BeliefInfo, queue_size=1)
        self.belief_info_msg = BeliefInfo()

        self.P_A_S_ALL_G_DICT = collections.OrderedDict()
        self.OPTIMAL_ACTION_FOR_S_G = []
        self.DEFAULT_PHI_GIVEN_A_NOISE = 0.1
        self.DEFAULT_PHM_GIVEN_PHI_NOISE = 0.1

        self.is_freeze_update = False
        self.eef_position = npa([0] * 3, dtype="f")
        self.eef_quat = npa([0] * 4, dtype="f")

        self.distribution_directory_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "jaco_personalized_distributions"
        )
        # unify the initialization of these distribution between different classes
        # init all distributions from file
        if os.path.exists(os.path.join(self.distribution_directory_path, str(self.subject_id) + "_p_phi_given_a.pkl")):
            print("LOADING PERSONALIZED P_PHI_GIVEN_A")
            with open(
                os.path.join(self.distribution_directory_path, str(self.subject_id) + "_p_phi_given_a.pkl"), "rb"
            ) as fp:
                self.P_PHI_GIVEN_A = pickle.load(fp)
        else:
            self.P_PHI_GIVEN_A = collections.OrderedDict()
            self.init_P_PHI_GIVEN_A()

        if os.path.exists(
            os.path.join(self.distribution_directory_path, str(self.subject_id) + "_p_phm_given_phi.pkl")
        ):
            print("LOADING PERSONALIZED P_PHM_GIVEN_PHI")
            with open(
                os.path.join(self.distribution_directory_path, str(self.subject_id) + "_p_phm_given_phi.pkl"), "rb"
            ) as fp:
                self.P_PHM_GIVEN_PHI = pickle.load(fp)
        else:
            self.P_PHM_GIVEN_PHI = collections.OrderedDict()
            self.init_P_PHM_GIVEN_PHI()

        rospy.loginfo("Waiting for sim_env node ")
        rospy.wait_for_service("/sim_env/get_prob_a_s_all_g")
        rospy.loginfo("sim_env node found!")
        self.get_prob_a_s_all_g = rospy.ServiceProxy("/sim_env/get_prob_a_s_all_g", PASAllG)

    def freeze_update(self, req):
        self.is_freeze_update = req.data
        response = SetBoolResponse()
        response.success = True
        return response

    def query_belief(self, req):
        # print("In Query belief")
        response = QueryBeliefResponse()
        self.lock.acquire()
        response.current_p_g_given_uh = list(self.P_G_GIVEN_UH)
        self.lock.release()
        return response

    def init_P_G_GIVEN_UH(self, req):
        """
        Initializes the p(g | phm) dict to uniform dictionary at the beginning of each trial
        """
        # service to be called at the beginning of each trial to reinit the distribution.
        # number of goals could be different for different goals.
        print("In Init Belief Service")
        self.NUM_GOALS = req.num_goals

        self.goal_positions = npa([[0] * 3] * self.NUM_GOALS, dtype="f")
        self.distanceToGoals = npa([100] * self.NUM_GOALS, dtype="f")
        self.delta_t = 0.1
        self.curr_pg = [1.0 / self.NUM_GOALS] * self.NUM_GOALS
        self.P_G_GIVEN_UH = npa([1.0 / self.NUM_GOALS] * self.NUM_GOALS)
        self.theta = npa([0] * self.NUM_GOALS, dtype="f")

        print("Initial Belief ", self.P_G_GIVEN_UH)
        response = InitBeliefResponse()
        response.status = True
        return response

    def reset_P_G_GIVEN_UH(self, req):
        self.NUM_GOALS = req.num_goals
        p_g_given_uh = req.p_g_given_uh
        assert len(p_g_given_uh) == self.NUM_GOALS
        self.P_G_GIVEN_UH = p_g_given_uh  # maybe do

        print("Reset Belief ", self.P_G_GIVEN_UH)
        response = ResetBeliefResponse()
        response.status = True
        return response

    def init_goal_locations(self, req):
        goal_loc_descs_list = req.obs_descs  # list of ObsDesc
        assert len(goal_loc_descs_list) == self.NUM_GOALS
        for i in range(self.NUM_GOALS):
            obs_position = goal_loc_descs_list[i].position  # Vector3
            self.goal_positions[i][0] = obs_position.x
            self.goal_positions[i][1] = obs_position.y
            self.goal_positions[i][2] = obs_position.z

        response = ObsDescListResponse()
        response.success = True
        print("Initialized goals", self.goal_positions)
        return response

    def update_current_position(self, position, orientation):
        self.eef_position = npa([position.x, position.y, position.z])
        self.eef_quat = npa([orientation.x, orientation.y, orientation.z, orientation.w])
        # print self.eef_position, self.eef_quat
        self.update_dist_theta()

    def update_dist_theta(self):
        for i in range(0, self.NUM_GOALS):
            self.distanceToGoals[i] = np.linalg.norm(self.eef_position - self.goal_positions[i])

    def compute_belief_update(self, intent_inference_data):
        self.lock.acquire()
        self.update_current_position(
            intent_inference_data.robot_pose.position, intent_inference_data.robot_pose.orientation
        )
        self.user_vel = intent_inference_data.user_vel.velocity.data  # 6D
        self.goal_velocities = intent_inference_data.robot_vels
        self.robot_discrete_state = intent_inference_data.robot_discrete_state  # (x,y,z,m) #where m is in 1,2,3,4,5,6
        # not needed for confidences because this is instantaneous
        self.P_G_GIVEN_UH = npa(intent_inference_data.current_p_g_given_uh)
        phm = intent_inference_data.phm
        current_mode = self.robot_discrete_state[-1]

        if not self.is_freeze_update:
            if current_mode in [1, 2, 3] and phm != "None" and phm != "input stopped":
                # do Bayesian update
                # print('IN BAYESIAN')
                p_a_s_all_g_response = self.get_prob_a_s_all_g()
                if p_a_s_all_g_response.status:
                    p_a_s_all_g = p_a_s_all_g_response.p_a_s_all_g
                    for g in range(len(p_a_s_all_g)):  # number of goals
                        self.P_A_S_ALL_G_DICT[g] = collections.OrderedDict()
                        for i, task_level_action in enumerate(TASK_LEVEL_ACTIONS):
                            self.P_A_S_ALL_G_DICT[g][task_level_action] = p_a_s_all_g[g].p_a_s_g[i]

                    # get optimal action for all goals for current state as a list ordered by goal index
                    self.OPTIMAL_ACTION_FOR_S_G = p_a_s_all_g_response.optimal_action_s_g
                    assert len(self.OPTIMAL_ACTION_FOR_S_G) == self.NUM_GOALS
                self._handle_bayesian_update(phm, current_mode)
            else:
                # print('IN DFT')
                # for rotation stuff
                dcdt = [0.0] * self.NUM_GOALS
                dcdt = npa(dcdt).reshape(self.NUM_GOALS, 1)
                if np.linalg.norm(self.user_vel) > 0.01:
                    tau = 4
                else:
                    tau = 5

                h = 1.0 / self.NUM_GOALS
                self.compute_current_input()
                l = -0.0 * np.ones((self.NUM_GOALS, self.NUM_GOALS))  # l is lambda
                np.fill_diagonal(l, 0)  # make diagonal elements zero
                l = 20 * np.eye(self.NUM_GOALS) + l
                l = np.matrix(l)
                TC = -(1 / tau) * np.eye(self.NUM_GOALS)

                dcdt = (
                    np.dot(TC, self.P_G_GIVEN_UH).reshape(self.NUM_GOALS, 1)
                    + (h / tau) * np.ones((self.NUM_GOALS, 1))
                    + np.asarray(l * self.sigmoid(self.curr_pg))
                )
                self.P_G_GIVEN_UH = (self.P_G_GIVEN_UH).reshape(self.P_G_GIVEN_UH.size, 1)
                # euler integration, could rk 4th order if I am too finicky about it.
                self.P_G_GIVEN_UH = self.P_G_GIVEN_UH + dcdt * self.delta_t
                # make all values always positive, can use np.finfo(np.double).tiny
                self.P_G_GIVEN_UH[self.P_G_GIVEN_UH <= 0] = 10 ** (-100)
                self.P_G_GIVEN_UH = self.P_G_GIVEN_UH / np.sum(self.P_G_GIVEN_UH)  # normalize sum = 1
                # self.P_G_GIVEN_UH = np.around(self.P_G_GIVEN_UH, 3)

        response = ComputeIntentResponse()
        response.updated_p_g_given_uh = list(self.P_G_GIVEN_UH)

        self.belief_info_msg.p_g_given_uh = list(self.P_G_GIVEN_UH)
        self.belief_info_pub.publish(self.belief_info_msg)
        self.lock.release()
        return response

    def compute_current_input(self):
        self.curr_pg = [0] * self.NUM_GOALS
        if np.linalg.norm(self.user_vel[:3]) > 0.001:
            for i in range(0, self.NUM_GOALS):
                # if np.linalg.norm(npa(list(self.goal_velocities[i].velocity.data[:3]))) > 0.001:
                normalized_vec = (self.goal_positions[i] - self.eef_position) / (
                    np.linalg.norm(self.goal_positions[i] - self.eef_position) + np.finfo(np.double).tiny
                )
                self.curr_pg[i] = np.dot(self.user_vel[:3], normalized_vec) / (
                    np.linalg.norm(self.user_vel[:3]) * np.linalg.norm(normalized_vec) + np.finfo(np.double).tiny
                )
                self.theta[i] = math.acos(self.curr_pg[i])
                self.curr_pg[i] = (self.curr_pg[i] + 1) * 0.5

        if np.linalg.norm(self.user_vel[3:6]) > 0.001:
            for i in range(0, self.NUM_GOALS):
                if np.linalg.norm(npa(list(self.goal_velocities[i].velocity.data[3:6]))) > 0.001:
                    # print "IN IF"
                    rot_rob_vel = self.goal_velocities[i].velocity.data[3:6]
                    # print rot_rob_vel
                    rot_align = np.dot(self.user_vel[3:6], rot_rob_vel) / (
                        np.linalg.norm(self.user_vel[3:6]) * np.linalg.norm(rot_rob_vel) + np.finfo(np.double).tiny
                    )
                    # print "ROT ALIGN", i, rot_rob_vel, rot_align
                    rot_align = (rot_align + 1) * 0.5
                    self.curr_pg[i] = self.curr_pg[i] + rot_align

        D = 0.17
        if np.linalg.norm(self.user_vel[:3]) > 0.01:
            for i in range(0, self.NUM_GOALS):
                if (
                    self.theta[i] < math.pi / 2.0
                ):  # if the direction of velocity is TOWARDS the goal and if it IS VERY NEAR A goal it meansthe user is going towrads the goal FO SURE
                    self.curr_pg[i] = self.curr_pg[i] + max(0, 1.0 - (self.distanceToGoals[i] / D))

        self.curr_pg = npa(self.curr_pg)
        # self.curr_pg = self.curr_pg/np.sum(self.curr_pg)
        self.curr_pg = self.curr_pg.reshape(self.curr_pg.size, 1)

    def _handle_bayesian_update(self, phm, current_mode):
        if phm != "None" and phm != "Soft-Hard Puff Deadband" and phm != "Soft-Hard Sip Deadband":
            for g in range(self.NUM_GOALS):
                likelihood = 0.0
                for a in self.P_PHI_GIVEN_A[current_mode].keys():
                    for phi in self.P_PHM_GIVEN_PHI.keys():
                        likelihood += (
                            self.P_PHM_GIVEN_PHI[phi][phm]
                            * self.P_PHI_GIVEN_A[current_mode][a][phi]
                            * self.P_A_S_ALL_G_DICT[g][a]
                        )
                self.P_G_GIVEN_UH[g] = self.P_G_GIVEN_UH[g] * likelihood  # multiply with prior

            # normalize
            self.P_G_GIVEN_UH = self.P_G_GIVEN_UH / np.sum(self.P_G_GIVEN_UH)

    def sigmoid(self, inp):  # helper function to compute sigmoid
        out = 1.0 / (1.0 + np.exp(-inp))
        out = out - 0.5
        return out

    def init_P_PHI_GIVEN_A(self):
        # only to be done at the beginning of a session for a subject. No updating between trials
        self.P_PHI_GIVEN_A = collections.OrderedDict()
        for mode in [1, 2, 3]:  # hard coded modes for XYZ of the JACO robot
            self.P_PHI_GIVEN_A[mode] = collections.OrderedDict()
            for k in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.keys():  # task level action
                self.P_PHI_GIVEN_A[mode][k] = collections.OrderedDict()
                for u in INTERFACE_LEVEL_ACTIONS:
                    if u == TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP[k]:
                        # try to weight the true command more for realistic purposes. Can be offset by using a high PHI_GIVEN_A_NOISE
                        self.P_PHI_GIVEN_A[mode][k][u] = 1.0
                    else:
                        self.P_PHI_GIVEN_A[mode][k][u] = 0.0

                delta_dist = np.array(list(self.P_PHI_GIVEN_A[mode][k].values()))
                uniform_dist = (1.0 / len(INTERFACE_LEVEL_ACTIONS)) * np.ones(len(INTERFACE_LEVEL_ACTIONS))
                blended_dist = (
                    1 - self.DEFAULT_PHI_GIVEN_A_NOISE
                ) * delta_dist + self.DEFAULT_PHI_GIVEN_A_NOISE * uniform_dist  # np.array
                for index, u in enumerate(INTERFACE_LEVEL_ACTIONS):
                    self.P_PHI_GIVEN_A[mode][k][u] = blended_dist[index]

    def init_P_PHM_GIVEN_PHI(self):
        """
        Generates a random p(um|ui). key = ui, subkey = um
        """
        self.P_PHM_GIVEN_PHI = collections.OrderedDict()
        for i in INTERFACE_LEVEL_ACTIONS:  # ui
            self.P_PHM_GIVEN_PHI[i] = collections.OrderedDict()
            for j in INTERFACE_LEVEL_ACTIONS:  # um
                if i == j:
                    # try to weight the true command more for realistic purposes. Can be offset by using a high UM_GIVEN_UI_NOISE
                    self.P_PHM_GIVEN_PHI[i][j] = 1.0
                else:
                    # P_PHM_GIVEN_PHI[i][j] = np.random.random()*UM_GIVEN_UI_NOISE#IF UM_GIVEN_UI_NOISE is 0, then the p(um|ui) is a deterministic mapping
                    self.P_PHM_GIVEN_PHI[i][j] = 0.0

            delta_dist = np.array(list(self.P_PHM_GIVEN_PHI[i].values()))
            uniform_dist = (1.0 / len(INTERFACE_LEVEL_ACTIONS)) * np.ones(len(INTERFACE_LEVEL_ACTIONS))
            blended_dist = (
                1 - self.DEFAULT_PHM_GIVEN_PHI_NOISE
            ) * delta_dist + self.DEFAULT_PHM_GIVEN_PHI_NOISE * uniform_dist  # np.array
            for index, j in enumerate(INTERFACE_LEVEL_ACTIONS):
                self.P_PHM_GIVEN_PHI[i][j] = blended_dist[index]


if __name__ == "__main__":
    JacoIntentInference()
    rospy.spin()
