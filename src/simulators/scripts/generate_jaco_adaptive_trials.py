#!/usr/bin/env python

import pickle
import collections
import itertools
import random
import argparse
import os

# from jaco_adaptive_assistance_utils import *

SCENES = collections.OrderedDict({'scene_1':4, 'scene_2':4})
ALGO_CONDITIONS = ['disamb', 'control']
HOME_POSITIONS = ['home_1', 'home_2']
TRANS_START_MODES = ['X', 'Y', 'Z'] 
ROT_TRANS_MODES = ['YAW', 'PITCH', 'ROLL']
MODES = TRANS_START_MODES + ROT_TRANS_MODES

TOTAL_TRIALS = 32
TRIALS_PER_ALGO = TOTAL_TRIALS / len(ALGO_CONDITIONS)
NUM_BLOCKS_PER_ALGO = 2
TRIALS_PER_BLOCK_PER_ALGO = TRIALS_PER_ALGO / NUM_BLOCKS_PER_ALGO


goals_for_scene = list(range(1, 5))
goals_for_scene = 4*goals_for_scene
print(goals_for_scene)
home_positions = 4*['home_1'] + 4*['home_2'] + 4*['home_1'] + 4*['home_2']
starting_modes = 2*MODES + random.sample(TRANS_START_MODES, 2) + random.sample(ROT_TRANS_MODES, 2)
random.shuffle(starting_modes)
print(starting_modes)
all_combinations_for_scenelist(zip(goals_for_scene, home_positions, starting_modes)))



def generate_experiment_trials(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial_dir",
        dest="trial_dir",
        default=os.path.join(os.getcwd(), "trial_folders", "trial_dir"),
        help="The directory where trials will be stored are",
    )
    parser.add_argument(
        "--metadata_dir",
        dest="metadata_dir",
        default=os.path.join(os.getcwd(), "trial_folders", "metadata_dir"),
        help="The directory where metadata of trials will be stored",
    )
    args = parser.parse_args()
    generate_experiment_trials(args)