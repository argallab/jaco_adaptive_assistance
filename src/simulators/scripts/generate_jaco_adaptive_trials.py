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



home_positions_1 = 6*['home_1']
goal_positions_1 = 3 * ['goal_3', 'goal_4']
random.shuffle(goal_positions_1)
print(goal_positions_1)
starting_modes_1 = TRANS_START_MODES + ROT_TRANS_MODES
random.shuffle(starting_modes_1)
print(starting_modes_1)
combo_h1 = list(zip(home_positions_1, goal_positions_1, starting_modes_1))

home_positions_2 = 6*['home_2']
goal_positions_2 = 3 * ['goal_1','goal_2']
random.shuffle(goal_positions_2)
print(goal_positions_2)
starting_modes_2 = TRANS_START_MODES + ROT_TRANS_MODES
random.shuffle(starting_modes_2)
print(starting_modes_2)
combo_h2 = list(zip(home_positions_2, goal_positions_2, starting_modes_2))

print('Disamb Scene 1\n')
#disamb block 1
random_3_from_combo_h1 = random.sample(range(6), 3)
random_3_from_combo_h2 = random.sample(range(6), 3)
disamb_block_0 = [combo_h1[i] for i in random_3_from_combo_h1] + [combo_h2[i] for i in random_3_from_combo_h2]

random.shuffle(disamb_block_0)
print(disamb_block_0)

print('Disamb Scene 2\n')
#disamb block 2

other_3_from_combo_h1 = [i for i in range(6) if i not in random_3_from_combo_h1]
other_3_from_combo_h2 = [i for i in range(6) if i not in random_3_from_combo_h2]
disamb_block_1 = [combo_h1[i] for i in other_3_from_combo_h1] + [combo_h2[i] for i in other_3_from_combo_h2]

random.shuffle(disamb_block_1)
print(disamb_block_1)

print('Control Scene 1\n')

#control block 1
random_3_from_combo_h1 = random.sample(range(6), 3)
random_3_from_combo_h2 = random.sample(range(6), 3)
control_block_0 = [combo_h1[i] for i in random_3_from_combo_h1] + [combo_h2[i] for i in random_3_from_combo_h2]

random.shuffle(control_block_0)
print(control_block_0)


#control block 2
print('Control Scene 2\n')


other_3_from_combo_h1 = [i for i in range(6) if i not in random_3_from_combo_h1]
other_3_from_combo_h2 = [i for i in range(6) if i not in random_3_from_combo_h2]
control_block_1 = [combo_h1[i] for i in other_3_from_combo_h1] + [combo_h2[i] for i in other_3_from_combo_h2]

random.shuffle(control_block_1)
print(control_block_1)

def generate_experiment_trials(args):
    pass
