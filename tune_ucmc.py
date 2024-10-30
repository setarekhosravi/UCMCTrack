#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    By using this code you can tune the trackers parameters with Optuna framework
    Attention: You should have TrackEval repository on your local machine, 
               then modify the pathes in this code.
    @author: STRH
    Created on Oct 28
"""

# import libraries
import os
import optuna
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# import tracker
from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
from track_to_eval import main

# running and evaluating tracker for every trials
def run_tracker(dataset, args, tracker):
    directories = [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
    directories.sort()

    directories = [item for item in directories if 'UAV' in item]

    for path in directories:
        args.input_path = os.path.join(dataset, f"{path}/img1")
        main(args= args, tracker= tracker)

def run_validation(args, tracker, trial_num):
    dataset = "/home/setare/Vision/Work/Tracking/Swarm Dataset/Drone Swarm/UAVSwarm-dataset/UAVSwarm-dataset-test"
    if not os.path.exists(os.path.join(dataset, "ucmctrack/data")):
        os.makedirs(os.path.join(dataset, "ucmctrack/data"))
    args.save_path = os.path.join(dataset, "ucmctrack/data")

    run_tracker(dataset, args, tracker)
    dataset_root = "/".join(dataset.split("/")[:-1])
    dataset_for_command = "\ ".join(dataset_root.split(" "))
    eval_command = f"python /home/setare/Vision/Work/Evaluation/TrackingEvaluation/TrackEval/scripts/run_mot_challenge.py \
        --GT_FOLDER {dataset_for_command} --TRACKERS_FOLDER {dataset_for_command} \
            --OUTPUT_FOLDER {dataset_for_command} --TRACKERS_TO_EVAL ucmctrack \
                --BENCHMARK UAVSwarm-dataset --SPLIT_TO_EVAL test"
    
    os.system(eval_command)
    
    result = os.path.join(dataset_root, "ucmctrack/pedestrian_detailed.csv")
    result_csv = pd.read_csv(result)

    HOTA = result_csv.at[26,'HOTA___AUC']
    MOTA = result_csv.at[26,'MOTA']
    MOTP = result_csv.at[26,'MOTP']
    IDF1 = result_csv.at[26,'IDF1']

    if not os.path.exists(os.path.join(dataset_root,"ucmc_trials")):
        os.mkdir(os.path.join(dataset_root,"ucmc_trials"))
    result_csv.to_csv(f"{dataset_root}/ucmc_trials/trial_{trial_num}.csv", encoding='utf-8', index=False)

    return HOTA, MOTA, MOTP, IDF1

def objective(trial):
    # Example: Replace with your actual evaluation code
    # dummy_metric = your_evaluation_function(track_thresh, track_buffer, match_thresh)
    a = trial.suggest_int('a', 1, 120)
    wx = trial.suggest_int('wx', 1, 10)
    wy = trial.suggest_int('wx', 1, 10)
    vmax = trial.suggest_int('vmax', 1, 20)
    max_age = trial.suggest_int('max_age', 0, 80)
    high_score = trial.suggest_float('high_score', 0, 1.0)
    
    # Initialize the tracker with these parameters
    tracker = UCMCTrack(a1=a, a2=a, wx=wx, wy=wy, 
                        vmax=vmax, max_age=max_age, 
                        fps=30, dataset="MOT", 
                        high_score=high_score,
                        use_cmc=False,
                        detector=None)

    HOTA, MOTA, MOTP, IDF1 =  run_validation(args, tracker=tracker, trial_num=trial.number)
    score = HOTA * 0.7 + MOTA * 0.1 + MOTP * 0.1 + IDF1 * 0.1
    return score

    # Return the metric to be optimized (minimized or maximized)
    # return dummy_metric

if __name__ == "__main__":
    # Set fixed arguments once at the beginning
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--cam_para', type=str, default = "demo/cam_para_test1.txt", help='camera parameter file name')
    parser.add_argument('--weights', type=str, help="detector weights path")
    parser.add_argument('--wx', type=float, default=5, help='wx')
    parser.add_argument('--wy', type=float, default=5, help='wy')
    parser.add_argument('--vmax', type=float, default=10, help='vmax')
    parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
    parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
    parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
    parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
    parser.add_argument('--input_type', type=str, required=True, help="Input type: image or video")
    parser.add_argument('--input_path', type=str, required=True, help="Path to input images folder or video file")
    parser.add_argument('--save_mot', type=str, required=True, help="save results in mot format")
    parser.add_argument('--save_path', type=str, required=False, help="path to folder for saving results")
    parser.add_argument('--gt', type=str, required=False, help="path to gt.txt file")

    args = parser.parse_args()

    study = optuna.create_study(study_name= "ucmctrack", storage= "sqlite:///ucmctrack.db", direction="maximize", load_if_exists=True)  # Choose "maximize" or "minimize"
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    print("Best Hyperparameters:", best_params)