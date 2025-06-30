# training/trainer.py

import torch
import torch.nn.functional as F
import numpy as np
import random
import time
import logging
import os
import sys
import signal
from datetime import datetime
from tqdm import tqdm

from .checkpoints import CheckpointManager
from .metrics import TrainingMetrics
from .self_play import SelfPlayManager


class Trainer:
    def __init__(self, model, optimizer, game, mcts, state_encoder, args, 
                 initial_state=None, log_dir="logs", use_mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.mcts = mcts
        self.state_encoder = state_encoder
        self.args = args
        self.initial_state = initial_state
        self.use_mixed_precision = use_mixed_precision
        
        # Initialize mixed precision scaler
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
            print("Using mixed precision training")
        else:
            self.scaler = None
            print("Using standard precision training")
        
        # Initialize logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{self.timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(model, optimizer, game, self.scaler, self.logger)
        self.metrics = TrainingMetrics(log_dir, self.timestamp, self.logger)
        self.self_play_manager = SelfPlayManager(game, mcts, state_encoder, self.logger)
        
        # Training state
        self.current_iteration = 0
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        
        self.logger.info("Enhanced AlphaZero Training Started")
        
    def graceful_shutdown(self, signum, frame):
        """Handle interruption gracefully"""
        self.logger.info(f"Received signal {signum}, saving checkpoint and shutting down...")
        self.checkpoint_manager.save_checkpoint(
            self.current_iteration, "interrupted_checkpoint", 
            self.metrics.training_history, self.args, self.timestamp
        )
        sys.exit(0)
    
    def train_on_batch(self, memory, iteration):
        """Train the model on a batch of experiences"""
        if len(memory) == 0:
            return 0, 0
            
        random.shuffle(memory)
        total_policy_loss = 0
        total_value_loss = 0
        batch_count = 0
        
        num_batches = (len(memory) + self.args['batch_size'] - 1) // self.args['batch_size']
        batch_pbar = tqdm(
            range(0, len(memory), self.args['batch_size']),
            desc=f"Iteration {iteration} Training",
            total=num_batches,
            leave=False,
            unit="batch"
        )
        
        for batch_idx in batch_pbar:
            batch_end = min(len(memory), batch_idx + self.args['batch_size'])
            sample = memory[batch_idx:batch_end]
            
            if len(sample) == 0:
                continue
                
            state, policy_targets, value_targets = zip(*sample)
            
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    out_policy, out_value = self.model(state)
                    log_probs = F.log_softmax(out_policy, dim=1)
                    policy_loss = -torch.sum(policy_targets * log_probs) / policy_targets.size(0)
                    value_loss = F.mse_loss(out_value, value_targets)
                    total_loss = policy_loss + value_loss
                
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out_policy, out_value = self.model(state)
                log_probs = F.log_softmax(out_policy, dim=1)
                policy_loss = -torch.sum(policy_targets * log_probs) / policy_targets.size(0)
                value_loss = F.mse_loss(out_value, value_targets)
                total_loss = policy_loss + value_loss
                
                total_loss.backward()
                self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batch_count += 1
            
            batch_pbar.set_postfix({
                'Policy Loss': f'{policy_loss.item():.4f}',
                'Value Loss': f'{value_loss.item():.4f}',
                'Total Loss': f'{total_loss.item():.4f}'
            })
        
        batch_pbar.close()
        return total_policy_loss / batch_count, total_value_loss / batch_count
    
    def train(self, model_dir, resume_from=None, use_parallel=False, num_processes=None):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.args['num_iterations']} iterations")
        self.logger.info(f"Model directory: {model_dir}")
        
        # Resume from checkpoint if specified
        start_iteration = 0
        if resume_from:
            checkpoint = self.checkpoint_manager.load_checkpoint(resume_from)
            if checkpoint:
                start_iteration = checkpoint['iteration'] + 1
                self.metrics.training_history = checkpoint['training_history']
                self.args.update(checkpoint['args'])
                self.logger.info(f"Resuming training from iteration {start_iteration}")
            else:
                self.logger.error("Failed to load checkpoint, starting from scratch")
        
        overall_start_time = time.time()
        self.metrics.start_time = overall_start_time
        
        if use_parallel:
            self.logger.info(f"Using parallel self-play with {num_processes or 'auto'} processes")
        
        main_pbar = tqdm(
            range(start_iteration, self.args['num_iterations']),
            desc="Training Progress",
            unit="iteration",
            position=0,
            initial=start_iteration,
            total=self.args['num_iterations']
        )
        
        for i in main_pbar:
            self.current_iteration = i
            iteration_start_time = time.time()
            memory = []
            self.model.eval()
            
            # Alternate starting player each iteration for balance
            starting_player = 1 if i % 2 == 0 else -1
            
            game_outcomes = {'p1_wins': 0, 'p_neg1_wins': 0, 'draws': 0}
            game_stats_list = []
            successful_games = 0
            
            # Self-play phase
            selfplay_start_time = time.time()
            
            if use_parallel:
                # Parallel self-play
                try:
                    results = self.self_play_manager.play_games_parallel(
                        self.args['num_self_play_iterations'],
                        starting_player,
                        i,
                        self.args,
                        self.initial_state,
                        num_processes
                    )
                    
                    # Process results
                    for game_memory, final_outcome_p1, _, game_stats in results:
                        if len(game_memory) > 0:
                            memory += game_memory
                            successful_games += 1
                            game_stats_list.append(game_stats)
                            
                            # Track game outcomes
                            if final_outcome_p1 == 1:
                                game_outcomes['p1_wins'] += 1
                            elif final_outcome_p1 == -1:
                                game_outcomes['p_neg1_wins'] += 1
                            else:
                                game_outcomes['draws'] += 1
                                
                    print(f"Parallel self-play completed: {successful_games}/{self.args['num_self_play_iterations']} games")
                    
                except Exception as e:
                    self.logger.error(f"Error in parallel self-play: {e}")
                    self.logger.info("Falling back to sequential self-play")
                    use_parallel = False
            
            if not use_parallel:
                # Sequential self-play (original implementation)
                selfplay_pbar = tqdm(
                    range(self.args['num_self_play_iterations']),
                    desc=f"Self-play (P{starting_player} starts)",
                    leave=False,
                    position=1,
                    unit="game"
                )
                
                for game_idx in selfplay_pbar:
                    try:
                        game_memory, final_outcome_p1, _, game_stats = self.self_play_manager.play_game(
                            starting_player, game_idx, iteration=i, args=self.args, initial_state=self.initial_state
                        )
                        
                        if len(game_memory) > 0:
                            memory += game_memory
                            successful_games += 1
                            game_stats_list.append(game_stats)
                            
                            # Track game outcomes
                            if final_outcome_p1 == 1:
                                game_outcomes['p1_wins'] += 1
                            elif final_outcome_p1 == -1:
                                game_outcomes['p_neg1_wins'] += 1
                            else:
                                game_outcomes['draws'] += 1
                            
                            selfplay_pbar.set_postfix({
                                'P1 Wins': game_outcomes['p1_wins'],
                                'P-1 Wins': game_outcomes['p_neg1_wins'],
                                'Draws': game_outcomes['draws'],
                                'Avg Moves': f"{np.mean([gs['moves'] for gs in game_stats_list]):.1f}",
                                'Samples': len(memory)
                            })
                                
                    except Exception as e:
                        self.logger.error(f"Error in self-play game {game_idx}: {e}")
                        continue
                
                selfplay_pbar.close()
            
            selfplay_time = time.time() - selfplay_start_time
            
            # Calculate game statistics
            total_games = sum(game_outcomes.values())
            if total_games == 0:
                self.logger.error("No successful games completed, skipping iteration")
                continue
                
            p1_win_rate = game_outcomes['p1_wins'] / total_games
            p_neg1_win_rate = game_outcomes['p_neg1_wins'] / total_games
            draw_rate = game_outcomes['draws'] / total_games
            
            avg_game_length = np.mean([gs['moves'] for gs in game_stats_list])
            avg_game_time = np.mean([gs['game_time'] for gs in game_stats_list])
            avg_mcts_per_move = np.mean([gs['avg_mcts_per_move'] for gs in game_stats_list])
            
            if len(memory) == 0:
                self.logger.warning("No training data collected, skipping training")
                continue
            
            # Training phase
            self.model.train()
            training_start_time = time.time()
            
            total_policy_loss = 0
            total_value_loss = 0
            
            epoch_pbar = tqdm(
                range(self.args['num_epochs']),
                desc="Training Epochs",
                leave=False,
                position=1,
                unit="epoch"
            )
            
            for epoch in epoch_pbar:
                policy_loss, value_loss = self.train_on_batch(memory, i)
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                
                epoch_pbar.set_postfix({
                    'Policy Loss': f'{policy_loss:.4f}',
                    'Value Loss': f'{value_loss:.4f}',
                    'Total Loss': f'{policy_loss + value_loss:.4f}'
                })
            
            epoch_pbar.close()
            training_time = time.time() - training_start_time
            
            avg_policy_loss = total_policy_loss / self.args['num_epochs']
            avg_value_loss = total_value_loss / self.args['num_epochs']
            avg_total_loss = avg_policy_loss + avg_value_loss
            
            # Learning rate decay
            current_lr = self.optimizer.param_groups[0]['lr']
            if i > 0 and i % self.args.get('lr_decay_steps', 15) == 0:
                old_lr = current_lr
                decay_factor = self.args.get('lr_decay_factor', 0.8)
                new_lr = old_lr * decay_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                current_lr = new_lr
                self.logger.info(f"LR decay: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Save model and checkpoint
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.model.state_dict(), f"{model_dir}/model_{i}.pt")
            torch.save(self.optimizer.state_dict(), f"{model_dir}/optimizer_{i}.pt")
            
            if i % 5 == 0 or i == self.args['num_iterations'] - 1:
                self.checkpoint_manager.save_checkpoint(
                    i, model_dir, self.metrics.training_history, self.args, self.timestamp
                )
            
            iteration_time = time.time() - iteration_start_time
            
            # Compile iteration statistics
            iteration_stats = {
                'iteration': i,
                'total_games': total_games,
                'p1_wins': game_outcomes['p1_wins'],
                'p_neg1_wins': game_outcomes['p_neg1_wins'],
                'draws': game_outcomes['draws'],
                'p1_win_rate': p1_win_rate,
                'p_neg1_win_rate': p_neg1_win_rate,
                'draw_rate': draw_rate,
                'avg_game_length': avg_game_length,
                'avg_game_time': avg_game_time,
                'avg_mcts_per_move': avg_mcts_per_move,
                'total_samples': len(memory),
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'total_loss': avg_total_loss,
                'learning_rate': current_lr,
                'selfplay_time': selfplay_time,
                'training_time': training_time,
                'total_time': iteration_time,
                'total_iterations': self.args['num_iterations']
            }
            
            # Update training history
            self.metrics.update_history(
                i, avg_policy_loss, avg_value_loss, avg_total_loss, current_lr,
                iteration_stats, selfplay_time, training_time, iteration_time
            )
            
            # Log summary
            self.metrics.log_iteration_summary(i, iteration_stats)
            
            # Update progress bar
            main_pbar.set_postfix({
                'P1 Win%': f'{p1_win_rate:.1%}',
                'Draws%': f'{draw_rate:.1%}',
                'Policy Loss': f'{avg_policy_loss:.4f}',
                'Value Loss': f'{avg_value_loss:.4f}',
                'LR': f'{current_lr:.6f}',
                'Time': f'{iteration_time:.1f}s'
            })
            
            # Save training history periodically
            if (i + 1) % 5 == 0:
                self.metrics.save_training_history(self.args)
        
        main_pbar.close()
        
        # Final training summary
        total_training_time = time.time() - overall_start_time
        self.logger.info(f"Training Complete! Total time: {total_training_time:.1f}s ({total_training_time/3600:.2f}h)")
        self.logger.info(f"Models saved to: {model_dir}")
        
        # Save final training history and checkpoint
        self.metrics.save_training_history(self.args)
        self.checkpoint_manager.save_checkpoint(
            self.args['num_iterations'] - 1, model_dir, 
            self.metrics.training_history, self.args, self.timestamp, is_best=True
        )