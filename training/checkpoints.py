# training/checkpoints.py

import torch
import numpy as np
import random
import os
import shutil


class CheckpointManager:
    def __init__(self, model, optimizer, game, scaler=None, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.scaler = scaler
        self.logger = logger
        
    def save_checkpoint(self, iteration, model_dir, training_history, args, timestamp, is_best=False):
        """Save complete training checkpoint"""
        os.makedirs(model_dir, exist_ok=True)
        
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': training_history,
            'args': args,
            'model_config': {
                'num_resBlocks': self.model.num_resBlocks,
                'num_hidden': self.model.num_hidden,
                'use_checkpoint': self.model.use_checkpoint
            },
            'game_config': {
                'row_count': self.game.row_count,
                'col_count': self.game.col_count,
                'buffer_count': self.game.buffer_count,
                'draw_move_limit': self.game.draw_move_limit,
                'repetition_limit': self.game.repetition_limit,
                'history_timesteps': self.game.history_timesteps
            },
            'timestamp': timestamp,
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state()
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save main checkpoint
        checkpoint_path = os.path.join(model_dir, f"checkpoint_{iteration}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # # Save latest checkpoint
        # latest_path = os.path.join(model_dir, "checkpoint_latest.pt")
        # torch.save(checkpoint, latest_path)
        
        # Save best model if specified
        if is_best:
            best_path = os.path.join(model_dir, "checkpoint_best.pt")
            shutil.copy2(checkpoint_path, best_path)
            if self.logger:
                self.logger.info(f"Saved best model at iteration {iteration}")
        
        if self.logger:
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint and resume"""
        if not os.path.exists(checkpoint_path):
            if self.logger:
                self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.model.device, weights_only=False)
            
            # Restore model and optimizer
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore random states for reproducibility
            if 'random_state' in checkpoint:
                random.setstate(checkpoint['random_state'])
            if 'numpy_random_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_random_state'])
            
            # Restore scaler if available
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            if self.logger:
                self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            
            return checkpoint
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading checkpoint: {e}")
            return None


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint file for inference"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None, None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None