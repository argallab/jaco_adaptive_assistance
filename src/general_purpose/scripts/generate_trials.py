#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

from utils import SCALE, VIEWPORT_W, VIEWPORT_H, ROBOT_RADIUS, GOAL_RADIUS, PI

import numpy as np
import math
import os
import pickle
from collections import OrderedDict

from IPython import embed

QUADRANT_BOUNDS = OrderedDict()
VIEWPORT_WS = VIEWPORT_W/SCALE
VIEWPORT_HS = VIEWPORT_H/SCALE
ROBOT_RADIUS_S = ROBOT_RADIUS/SCALE
GOAL_RADIUS_S = GOAL_RADIUS/SCALE
R_TO_G_CONFIGS = {'tr':{'r':'3', 'g': '1'},
                  'tl':{'r':'4', 'g': '2'},
                  'br':{'r':'2', 'g': '4'},
                  'bl':{'r':'1', 'g': '3'}}
R_TO_G_ORIENT_DIFF = PI/2
NUM_TURNS = [1,2,3]

def create_bounds_dict():
    q_keys = [1,2,3,4]
    dimensions = ['x', 'y']
    bounds = ['min', 'max']
    for q_key in q_keys:
        QUADRANT_BOUNDS[str(q_key)] = OrderedDict()
        for d in dimensions:
            QUADRANT_BOUNDS[str(q_key)][d] = OrderedDict()
            for b in bounds:
                QUADRANT_BOUNDS[str(q_key)][d][b] = None


def initialize_bounds():
    #initialize bounds for 'tr'
    QUADRANT_BOUNDS['1']['x']['min'] = (2*VIEWPORT_WS)/3 + ROBOT_RADIUS_S
    QUADRANT_BOUNDS['1']['x']['max'] = VIEWPORT_WS - ROBOT_RADIUS_S
    QUADRANT_BOUNDS['1']['y']['min'] = (2*VIEWPORT_HS)/3 + ROBOT_RADIUS_S
    QUADRANT_BOUNDS['1']['y']['max'] = VIEWPORT_HS - ROBOT_RADIUS_S

    #initalize bounds for 'tl'
    QUADRANT_BOUNDS['2']['x']['min'] = ROBOT_RADIUS_S
    QUADRANT_BOUNDS['2']['x']['max'] = VIEWPORT_WS/3 - ROBOT_RADIUS_S
    QUADRANT_BOUNDS['2']['y']['min'] = (2*VIEWPORT_HS)/3 + ROBOT_RADIUS_S
    QUADRANT_BOUNDS['2']['y']['max'] = VIEWPORT_HS - ROBOT_RADIUS_S

    #initialize_bounds for 'bl'
    QUADRANT_BOUNDS['3']['x']['min'] = ROBOT_RADIUS_S
    QUADRANT_BOUNDS['3']['x']['max'] = VIEWPORT_WS/3 - ROBOT_RADIUS_S
    QUADRANT_BOUNDS['3']['y']['min'] = ROBOT_RADIUS_S
    QUADRANT_BOUNDS['3']['y']['max'] = VIEWPORT_HS/3 - ROBOT_RADIUS_S

    #initialize_bounds for 'br'
    QUADRANT_BOUNDS['4']['x']['min'] = (2*VIEWPORT_WS)/3 + ROBOT_RADIUS_S
    QUADRANT_BOUNDS['4']['x']['max'] = VIEWPORT_WS - ROBOT_RADIUS_S
    QUADRANT_BOUNDS['4']['y']['min'] = ROBOT_RADIUS_S
    QUADRANT_BOUNDS['4']['y']['max'] = VIEWPORT_HS/3 - ROBOT_RADIUS_S

def create_r_to_g_configurations(num_trials):
    r_to_g_config_list = []
    num_unique_configs = len(R_TO_G_CONFIGS.keys())
    for key in R_TO_G_CONFIGS.keys():
        r_to_g_config_list.extend([key]*(num_trials//num_unique_configs))

    return r_to_g_config_list

def generate_r_and_g_positions(num_trials, r_to_g_config_list):
    robot_positions = np.zeros((num_trials,2))
    goal_positions = np.zeros((num_trials, 2))
    for i, rgc in enumerate(r_to_g_config_list):
        rq = R_TO_G_CONFIGS[rgc]['r']
        gq = R_TO_G_CONFIGS[rgc]['g']
        rx = QUADRANT_BOUNDS[rq]['x']['min'] + np.random.random()*(QUADRANT_BOUNDS[rq]['x']['max']-QUADRANT_BOUNDS[rq]['x']['min'])
        ry = QUADRANT_BOUNDS[rq]['y']['min'] + np.random.random()*(QUADRANT_BOUNDS[rq]['y']['max']-QUADRANT_BOUNDS[rq]['y']['min'])
        gx = QUADRANT_BOUNDS[gq]['x']['min'] + np.random.random()*(QUADRANT_BOUNDS[gq]['x']['max']-QUADRANT_BOUNDS[gq]['x']['min'])
        gy = QUADRANT_BOUNDS[gq]['y']['min'] + np.random.random()*(QUADRANT_BOUNDS[gq]['y']['max']-QUADRANT_BOUNDS[gq]['y']['min'])
        robot_positions[i] = (rx, ry)
        goal_positions[i] = (gx, gy)

    return robot_positions, goal_positions

def generate_r_and_g_orientations(num_trials):
    robot_init_orientations = np.zeros((num_trials, 1))
    goal_orientations = np.zeros((num_trials, 1))
    for i in range(num_trials):
        robot_init_orientations[i] = np.random.random()*(2*PI)
        if np.random.random() < 0.5:
            goal_orientations[i] = robot_init_orientations[i] + R_TO_G_ORIENT_DIFF
        else:
            goal_orientations[i] = robot_init_orientations[i] - R_TO_G_ORIENT_DIFF

    return robot_init_orientations, goal_orientations

def generate_num_turns_list(num_trials):
    num_turns_trial_list = []
    for num_turn in NUM_TURNS:
        num_turns_trial_list.extend([num_turn]*(num_trials/len(NUM_TURNS)))

    # TODO (FIX THISSSS)
    return num_turns_trial_list


def generate_trials(num_trials=48):
    assert num_trials%len(NUM_TURNS) == 0
    assert num_trials%len(R_TO_G_CONFIGS.keys()) == 0

    create_bounds_dict()
    initialize_bounds()
    r_to_g_config_list = create_r_to_g_configurations(num_trials)
    num_trials = len(r_to_g_config_list)
    robot_init_positions, goal_positions = generate_r_and_g_positions(num_trials, r_to_g_config_list)
    robot_init_orientations, goal_orientations = generate_r_and_g_orientations(num_trials)
    num_turns_list = generate_num_turns_list(num_trials)

if __name__ == '__main__':
    generate_trials()
