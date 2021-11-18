#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import os
import csv
import sys
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import itertools
import collections
import bisect
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
from jaco_adaptive_assistance_utils import *


class DataParser(object):
    def __init__(self, trans_file_dir, modes_file_dir):
        super(DataParser, self).__init__()

        trans_results_files = os.listdir(trans_file_dir)
        trans_action_prompt_file = os.path.join(trans_file_dir, "_action_prompt.csv")
        trans_user_response_file = os.path.join(trans_file_dir, "_user_response.csv")

        modes_results_files = os.listdir(modes_file_dir)
        modes_action_prompt_file = os.path.join(modes_file_dir, "_action_prompt.csv")
        modes_user_response_file = os.path.join(modes_file_dir, "_user_response.csv")

        self.trans_action_prompt_df = self.read_csv_files(trans_action_prompt_file)
        self.trans_user_response_df = self.read_csv_files(trans_user_response_file)

        self.modes_action_prompt_df = self.read_csv_files(modes_action_prompt_file)
        self.modes_user_response_df = self.read_csv_files(modes_user_response_file)

    def read_csv_files(self, file_path):
        df = pd.read_csv(file_path, header=0)
        return df


class PhiGivenAAnalysis(object):
    def __init__(self, args):

        self.id = args.id
        self.trans_file_dir = os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "raw_data", self.id + "trans_p_phi_given_a"
        )
        self.modes_file_dir = os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "raw_data", self.id + "modes_p_phi_given_a"
        )
        # if not os.path.exists(self.trans_file_dir):
        #     os.makedirs(self.trans_file_dir)

        self.data = DataParser(self.trans_file_dir, self.modes_file_dir)

    def get_nearest_time_stamp(self, tq, time_stamp_array):
        """
        assumes that time_stamp_array is a sorted (ascending) list.
        returns: (time stamp closest to tq, index of the time stamp closest to tq in time_stamp_array)
        """
        # idx is the index in the time_stamp_array whose value is immediately greater than tq
        idx = bisect.bisect(time_stamp_array, tq)
        # if requested timestamp is smaller than the first element of the list then return the first element of the list and the index is 0
        if idx == 0:
            return time_stamp_array[idx], idx
        elif idx == len(time_stamp_array):
            # if tq is greater than the biggest timestamp in the list, then retirn the last timestamp and the index of len-1 or idx-1
            return time_stamp_array[idx - 1], idx - 1
        else:
            prev_t = time_stamp_array[idx - 1]
            next_t = time_stamp_array[idx]
            if next_t - tq <= tq - prev_t:  # if tq is closer to the next value
                return time_stamp_array[idx], idx
            else:  # tq is closer to the previous value
                return time_stamp_array[idx - 1], idx - 1

    def get_user_response_block_indices(self, time_s, time_e, user_input):

        assert "rosbagTimestamp" in user_input
        time_s_u, index_of_time_s_u = self.get_nearest_time_stamp(time_s, user_input.rosbagTimestamp)
        time_e_u, index_of_time_e_u = self.get_nearest_time_stamp(time_e, user_input.rosbagTimestamp)

        if index_of_time_e_u == index_of_time_s_u:  # sanity check
            user_response_block_indices = index_of_time_s_u

        elif time_s_u < time_s and time_e_u < time_e:
            user_response_block_indices = index_of_time_e_u

        elif time_s_u > time_s and time_e_u > time_e:
            user_response_block_indices = index_of_time_s_u

        else:
            user_response_block_indices = []  # no user response (it's from previous)

        return user_response_block_indices

    def build_distributions(self):

        # create dictionary to map user input to the low level commands for the final p_ui_given_a dict
        USER_RESPONSE_DICT = {"1": "Hard Puff", "2": "Hard Sip", "3": "Soft Puff", "4": "Soft Sip"}

        # hard puff, hard sip, soft puff, soft sip
        right = np.zeros(4)
        left = np.zeros(4)
        forward = np.zeros(4)
        backward = np.zeros(4)
        down = np.zeros(4)
        up = np.zeros(4)
        mode_r_x = np.zeros(4)
        mode_r_y = np.zeros(4)
        mode_r_z = np.zeros(4)
        mode_l_x = np.zeros(4)
        mode_l_y = np.zeros(4)
        mode_l_z = np.zeros(4)

        # because there are x2 as many prmpts (the actual command and the empty command after)
        # and 12 total actions, so the remaining is the number of times each action was shown since all shown equally
        iters_per_action = len(self.data.trans_action_prompt_df) / 24
        print("ITER PER ACTION FOR TRANS", iters_per_action)
        print("ITER PER ACTION FOR MODES", len(self.data.modes_action_prompt_df) / 24)

        # dictionary for mapping action prompts to the arrays we want to fill
        # TODO update the keys and then later combine fields
        ACTION_TO_ARRAY_DICT = {
            "right": right,
            "left": left,
            "forward": forward,
            "backward": backward,
            "down": down,
            "up": up,
            "mode_switch_right_1": mode_r_x,
            "mode_switch_right_2": mode_r_y,
            "mode_switch_right_3": mode_r_z,
            "mode_switch_left_1": mode_l_x,
            "mode_switch_left_2": mode_l_y,
            "mode_switch_left_3": mode_l_z,
        }

        # keep dict of lengths, to reduce normalizer if person missed input
        ACTION_TO_ARRAY_NORMALIZER_DICT = {
            "right": iters_per_action,
            "left": iters_per_action,
            "forward": iters_per_action,
            "backward": iters_per_action,
            "down": iters_per_action,
            "up": iters_per_action,
            "mode_switch_right_1": iters_per_action,
            "mode_switch_right_2": iters_per_action,
            "mode_switch_right_3": iters_per_action,
            "mode_switch_left_1": iters_per_action,
            "mode_switch_left_2": iters_per_action,
            "mode_switch_left_3": iters_per_action,
        }

        # change all c2_* to c1_* for ease of parsing in self.data.trans
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c1_right"', "command"
        ] = '"right"'
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c1_left"', "command"
        ] = '"left"'
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c1_forward"', "command"
        ] = '"forward"'
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c1_backward"', "command"
        ] = '"backward"'
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c1_down"', "command"
        ] = '"down"'
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c1_up"', "command"
        ] = '"up"'

        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c2_right"', "command"
        ] = '"right"'
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c2_left"', "command"
        ] = '"left"'
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c2_forward"', "command"
        ] = '"forward"'
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c2_backward"', "command"
        ] = '"backward"'
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c2_down"', "command"
        ] = '"down"'
        self.data.trans_action_prompt_df.loc[
            self.data.trans_action_prompt_df["command"] == '"c2_up"', "command"
        ] = '"up"'

        # for every trans action prompt, get the user response (if they did respond)
        for i in range(0, len(self.data.trans_action_prompt_df) - 1, 2):

            key = self.data.trans_action_prompt_df.at[i, "command"].replace('"', "")

            prompt_t_s = self.data.trans_action_prompt_df.at[i, "rosbagTimestamp"]
            prompt_t_e = self.data.trans_action_prompt_df.at[i + 1, "rosbagTimestamp"]

            user_response_block_indices = self.get_user_response_block_indices(
                prompt_t_s, prompt_t_e, self.data.trans_user_response_df
            )

            if user_response_block_indices != []:  # if they gave a response
                user_response = int(
                    self.data.trans_user_response_df["command"][user_response_block_indices].replace('"', "")
                )
                ACTION_TO_ARRAY_DICT[key][user_response - 1] += 1
            else:
                ACTION_TO_ARRAY_NORMALIZER_DICT[key] -= 1

        # for every mode switch action prompt, get the user response (if they did respond)
        for i in range(0, len(self.data.modes_action_prompt_df) - 1, 2):

            key = self.data.modes_action_prompt_df.at[i, "command"].replace('"', "")
            if key not in [
                "mode_switch_right_1",
                "mode_switch_right_2",
                "mode_switch_right_3",
                "mode_switch_left_1",
                "mode_switch_left_2",
                "mode_switch_left_3",
            ]:
                continue

            prompt_t_s = self.data.modes_action_prompt_df.at[i, "rosbagTimestamp"]
            prompt_t_e = self.data.modes_action_prompt_df.at[i + 1, "rosbagTimestamp"]

            user_response_block_indices = self.get_user_response_block_indices(
                prompt_t_s, prompt_t_e, self.data.modes_user_response_df
            )

            if user_response_block_indices != []:  # if they gave a response
                user_response = int(
                    self.data.modes_user_response_df["command"][user_response_block_indices].replace('"', "")
                )
                ACTION_TO_ARRAY_DICT[key][user_response - 1] += 1
            else:
                ACTION_TO_ARRAY_NORMALIZER_DICT[key] -= 1

        # normalize the dict which contains both trans and modes info. only concerned about performance in trans actions and mode switches in trans modes
        for k, v in ACTION_TO_ARRAY_DICT.items():
            v = v / ACTION_TO_ARRAY_NORMALIZER_DICT[k]
            ACTION_TO_ARRAY_DICT[k] = v

        self.create_p_phi_given_a(ACTION_TO_ARRAY_DICT)

    def create_p_phi_given_a(self, probabilities):
        TRUE_P_PHI_GIVEN_A = collections.OrderedDict()
        modes = [1, 2, 3]
        for m in modes:
            TRUE_P_PHI_GIVEN_A[m] = collections.OrderedDict()
            for a in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.keys():
                TRUE_P_PHI_GIVEN_A[m][a] = collections.OrderedDict()
                for phi in INTERFACE_LEVEL_ACTIONS:
                    if phi == TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP[a]:
                        TRUE_P_PHI_GIVEN_A[m][a][phi] = 1.0
                    else:
                        TRUE_P_PHI_GIVEN_A[m][a][phi] = 0.0
        keys = ["Hard Puff", "Hard Sip", "Soft Puff", "Soft Sip"]
        p_phi_given_a = collections.OrderedDict()
        diff_norm_p_phi_given_a = collections.OrderedDict()
        for mode in modes:
            p_phi_given_a[mode] = collections.OrderedDict()
            diff_norm_p_phi_given_a[mode] = collections.OrderedDict()
            for action in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.keys():
                p_phi_given_a[mode][action] = collections.OrderedDict()

                if mode == 1:
                    if action == "move_p":
                        prob = probabilities["right"]  # each pf these a 4D array
                    if action == "move_n":
                        prob = probabilities["left"]
                    if action == "to_mode_r":
                        prob = probabilities["mode_switch_right_1"]
                    if action == "to_mode_l":
                        prob = probabilities["mode_switch_left_1"]  # not relevant for jaCO MDP
                if mode == 2:
                    if action == "move_p":
                        prob = probabilities["forward"]
                    if action == "move_n":
                        prob = probabilities["backward"]
                    if action == "to_mode_r":
                        prob = probabilities["mode_switch_right_2"]
                    if action == "to_mode_l":
                        prob = probabilities["mode_switch_left_2"]
                if mode == 3:
                    if action == "move_p":
                        prob = probabilities["down"]
                    if action == "move_n":
                        prob = probabilities["up"]
                    if action == "to_mode_r":
                        prob = probabilities["mode_switch_right_3"]  # not relevant for JACO MDP
                    if action == "to_mode_l":
                        prob = probabilities["mode_switch_left_3"]

                noise_level = 0.0001
                prob = prob + noise_level * np.ones((4,))
                prob = prob / np.sum(prob)

                for ind, key in enumerate(keys):
                    p_phi_given_a[mode][action][key] = prob[ind]

                diff_norm_p_phi_given_a[mode][action] = np.linalg.norm(
                    np.array(TRUE_P_PHI_GIVEN_A[mode][action].values()) - prob
                )

                flatten = itertools.chain.from_iterable
                list_of_all_diff_norms = [d.values() for d in diff_norm_p_phi_given_a.values()]
                list_of_all_diff_norms = list(flatten(list_of_all_diff_norms))

        print(max(list_of_all_diff_norms))
        if max(list_of_all_diff_norms) > 0.22:
            print("NEEEEDS MORE TRAINING", max(list_of_all_diff_norms))

        personalized_distributions_dir = os.path.join(
            rospkg.RosPack().get_path("jaco_intent_inference"), "personalized_distributions"
        )
        if not os.path.exists(personalized_distributions_dir):
            os.makedirs(personalized_distributions_dir)
        pickle.dump(
            p_phi_given_a, open(os.path.join(personalized_distributions_dir, self.id + "_p_phi_given_a.pkl"), "wb")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", help="subject id", type=str)
    args = parser.parse_args()
    pphia = PhiGivenAAnalysis(args)
    pphia.build_distributions()
