"""Shared runtime helpers for configuration, scaling, and checkpoint logic.

The training scripts import these helpers to keep argument parsing, scaler
construction, checkpoint discovery, and relaxation worker management unified.
"""

import os
import numpy as np
import torch
import argparse
import subprocess
import sys
import os
from utils.set_logger import get_logger
from utils.config import DictAction, get_params


def dict2namespace(config):
    """Recursively convert a nested mapping into an argparse namespace."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config(task):
    """Parse CLI arguments, load the task config, and build runtime state."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mp_20')
    parser.add_argument('--port', type=str, default='11111')
    parser.add_argument('--conf_new', nargs='+', action=DictAction)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_seed', action='store_true')
    args = parser.parse_args()
    
    args.task = task
    if task != "jepa":
        args.log = os.path.join('logs', args.task, args.dataset)
        args.cfg = os.path.join('configs', args.task, args.dataset+'.yml')
    else:
        args.log = os.path.join('logs', args.task)
        args.cfg = os.path.join('configs', args.task, 'mp.yml')

    os.makedirs(args.log, exist_ok=True)
    config = get_params(args)
    new_config = dict2namespace(config)

    #### setup logger #### 
    logger = get_logger(path=os.path.join(args.log, 'running.log'))
    args.logger = logger

    # add device
    device = torch.device('cuda')
    logger.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    if args.use_seed == True:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

class Scaler_min_max(object):
    def __init__(self, min=None, max=None):
        """Store min/max values for simple linear rescaling."""
        self.min = min
        self.max = max

    def fit(self, X):
        """Estimate the min and max values from a tensor."""
        X = X.float()
        self.min = torch.min(X)
        self.max = torch.max(X)

    def transform(self, X):
        """Scale a tensor into the [0, 1] interval."""
        X = X.float()
        new_X = (X - self.min.to(X.device)) / (self.max.to(X.device) - self.min.to(X.device))
        return new_X

    def inverse_transform(self, X):
        """Map a normalized tensor back to the original value range."""
        X = X.float()
        new_X = X * (self.max.to(X.device) - self.min.to(X.device)) + self.min.to(X.device)
        return new_X

    def pack(self):
        """Serialize the scaler state into a tensor."""
        return torch.tensor([self.min, self.max])
    
    def load(self, re):
        """Restore the scaler state from a packed tensor."""
        self.min, self.max = re

class Scaler_mean_std(object):
    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """Store mean/std statistics for standardization."""
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """Estimate feature-wise mean and standard deviation."""
        self.means = torch.mean(X, dim=0, keepdim=True)
        self.stds = torch.std(X, dim=0, keepdim=True)
        return self

    def transform(self, X):
        """Standardize a tensor using the stored statistics."""
        X = X.float()
        new_X = (X - self.means.to(X.device)) / self.stds.to(X.device)
        return new_X

    def inverse_transform(self, X):
        """Undo the standardization transform."""
        X = X.float()
        new_X = X * self.stds.to(X.device) + self.means.to(X.device)
        return new_X

    def pack(self):
        """Serialize mean/std values into a tensor."""
        return torch.cat([self.means, self.stds], 0)
    
    def load(self, re):
        """Restore the mean/std values from a packed tensor."""
        self.means, self.stds = re

def get_scaler_min_max(task, dataset, scaled_matrix=None):
    """Load or fit the min-max scaler for a task dataset."""
    matrix_scaler_path = os.path.join('data', task, dataset, 'min_max_scaler.pt')
    matrix_scaler = Scaler_min_max()
    if not os.path.exists(matrix_scaler_path):
        matrix_scaler.fit(scaled_matrix)
        torch.save(matrix_scaler.pack(), matrix_scaler_path)
    else:
        matrix_scaler.load(torch.load(matrix_scaler_path))
    return matrix_scaler

def get_scaler_mean_std(task, dataset=None, scaled_matrix=None):
    """Load or fit the mean/std scaler for a task dataset."""
    if dataset is None:
        matrix_scaler_path = os.path.join('data', task, 'mean_std_scaler.pt')
    else:
        matrix_scaler_path = os.path.join('data', task, dataset, 'mean_std_scaler.pt')
    matrix_scaler = Scaler_mean_std()
    if not os.path.exists(matrix_scaler_path):
        matrix_scaler.fit(scaled_matrix)
        torch.save(matrix_scaler.pack(), matrix_scaler_path)
    else:
        matrix_scaler.load(torch.load(matrix_scaler_path))
    return matrix_scaler

def last_ckpt(task, dataset=None):
    """Return the newest checkpoint path for a task and optional dataset."""
    if dataset is None:
        saved_model_path = os.path.join('logs', task, 'saved_model')
    else:
        saved_model_path = os.path.join('logs', task, dataset, 'saved_model')
    all_files = os.listdir(saved_model_path)
    all_saved_epochs = []
    for file in all_files:
        if os.path.splitext(file)[1]==".pt":
            all_saved_epochs.append(int(file.split('.')[0].split('_')[1]))
    all_saved_epochs.sort()
    ckpt = os.path.join(saved_model_path, 'model_'+str(all_saved_epochs[-1])+'.pt')
    return ckpt

def check_save_num(save_path):
    """Keep only the ten most recent checkpoint files in a directory."""
    all_files = os.listdir(save_path)
    all_saved_epochs, all_saved_files = [], []
    for file in all_files:
        if os.path.splitext(file)[1]==".pt":
            all_saved_files.append(str(file))
            all_saved_epochs.append(int(os.path.splitext(file)[0].split("_")[1]))
    if len(all_saved_epochs) > 10:
        all_saved_files, all_saved_epochs = np.array(all_saved_files), np.array(all_saved_epochs)
        idx = np.argsort(all_saved_epochs)
        all_saved_files, all_saved_epochs = all_saved_files[idx], all_saved_epochs[idx]
        remove_files = all_saved_files[:-10]
        for ff in remove_files:
            os.remove(os.path.join(save_path, ff))

def run_relax_subprocess(turn, args):
    """Run the relaxation worker for one generation turn and load results."""
    cmd = [
        sys.executable,
        "relax_worker.py",
        "--gen_path", os.path.join(args.gen_path, f"gen_{turn}.pt"),
        "--relax_path", os.path.join(args.eval_path, str(turn), f"relaxed_{args.mlff}.json"),
        "--eval_path", os.path.join(args.eval_path, str(turn)),
        "--mlff", args.mlff,
        "--steps", str(args.max_relax_steps),
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Relax subprocess failed for turn={turn}")
    
    v_indicator = np.load(os.path.join(args.eval_path, str(turn), "v_indicator.npy"))
    return len(v_indicator), v_indicator
