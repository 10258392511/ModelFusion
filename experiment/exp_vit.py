import sys
sys.path.append('/cluster/project/infk/cvg/students/junwang/')
sys.path.append('/cluster/project/infk/cvg/students/junwang/ModelFusion')
sys.path.append('/')

import argparse
import subprocess
import numpy as np
import os

from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_FILENAME = os.path.join(ROOT_DIR, "scripts/evaluate_fused_vit.py")
RETRAIN_SCRIPTS_FILENAME = os.path.join(ROOT_DIR, "scripts/retrain_fused_vit.py")


def create_command(hyper_param_dict: dict, exp_num):
    # root
    if exp_num == 1 or exp_num == 2:
        command = f"""
        python {SCRIPTS_FILENAME} --ensemble_step {hyper_param_dict["ensemble_step"]} --square_factor {hyper_param_dict["square_factor"]} --retrain_fraction {hyper_param_dict["retrain_fraction"]}
        """
    elif exp_num == 3:
        command = f"""
        python {RETRAIN_SCRIPTS_FILENAME} --ensemble_step {hyper_param_dict["ensemble_step"]} --square_factor {hyper_param_dict["square_factor"]} --retrain_fraction {hyper_param_dict["retrain_fraction"]}
        """
    elif exp_num == 4:
        command = f"""
        python {RETRAIN_SCRIPTS_FILENAME} --average --ensemble_step {hyper_param_dict["ensemble_step"]} --square_factor {hyper_param_dict["square_factor"]} --retrain_fraction {hyper_param_dict["retrain_fraction"]}
        """

    return command


def get_hyper_params(set_num: int):
    hyper_params = defaultdict(list)
    # set_num = 1: experimenting on "ensemble_step"
    for ensemble_step_iter in np.arange(0., 1.01, 0.1):
        hyper_params[1].append({
            "ensemble_step": ensemble_step_iter,
            "square_factor": "1/8",
            "retrain_fraction": 1.,
            # "save_dir": "./logs"
        })

    # set_num = 2: experimenting on "square_factor"
    # for den_iter in [0.5, 1, 2, 5, 8, 10, 20]:
    for den_iter in [1/5]:
        hyper_params[2].append({
            "ensemble_step": 0.5,
            "square_factor": f"1/{den_iter}",
            "retrain_fraction": 1.,
            # "save_dir": "./logs"
        })

    # set_num = 3: experimenting on "retrain percent"
    for retrain_frac_iter in [0.1, 0.3, 0.5, 0.75, 1.]:
        hyper_params[3].append({
            "ensemble_step": 0.7,
            "square_factor": "1/2",
            "retrain_fraction": retrain_frac_iter,
            # "save_dir": "./logs"
        })

    # set_num = 4: experimenting on "retrain percent"
    for retrain_frac_iter in [0.1, 0.3, 0.5, 0.75, 1.]:
        hyper_params[4].append({
            "ensemble_step": 0.7,
            "square_factor": "1/2",
            "retrain_fraction": retrain_frac_iter,
            # "save_dir": "./logs"
        })

    return hyper_params[set_num]


if __name__ == '__main__':
    """
    python dev/exp_vit.py --set_num 1 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_num", type=int, required=True)
    args = vars(parser.parse_args())

    hyper_params = get_hyper_params(args["set_num"])
    for hyper_param_dict in hyper_params:
        command = create_command(hyper_param_dict, args["set_num"])
        subprocess.call(command, shell=True)
