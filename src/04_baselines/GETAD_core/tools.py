"""
GETAD_core/tools.py
Logging and early stopping utilities for the GETAD baseline.
"""

import logging
from pathlib import Path
import sys
import torch


def setup_logger(args, current_time=None, log_dir="../logs"):
    """
    Sets up formatted stdout and file logger.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    time_str = current_time or ""
    log_filename = f"getad_{args.region}_{args.tag}_{time_str}.log"
    log_file = log_dir_path / log_filename

    logger = logging.getLogger(f"GETAD_{args.region}_{args.tag}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class EarlyStopping:
    """
    Monitors validation loss and saves the best model checkpoint.
    """
    def __init__(self, logger=None, patience=5, delta=0.0, verbose=True):
        self.logger = logger
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_checkpoint_path = None

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        elif self.verbose:
            print(msg)

    def __call__(self, val_loss, model, ckpt_dir, ckpt_filename):
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_path = ckpt_dir / ckpt_filename

        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(None, val_loss, model, save_path)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            self._log(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            prev_loss = self.best_loss
            self.best_loss = val_loss
            self._save_checkpoint(prev_loss, val_loss, model, save_path)
            self.counter = 0

    def _save_checkpoint(self, prev_loss, val_loss, model, save_path):
        if prev_loss is None:
            self._log(f"Initial validation loss ({val_loss:.6f}). Saving initial model...")
        else:
            self._log(f"Validation loss decreased ({prev_loss:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), save_path)
        self.best_checkpoint_path = save_path
