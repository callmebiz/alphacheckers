# training/metrics.py

import json
import time
import os


class TrainingMetrics:
    def __init__(self, log_dir, timestamp, logger=None):
        self.log_dir = log_dir
        self.timestamp = timestamp
        self.logger = logger
        self.start_time = time.time()
        
        self.training_history = {
            'iterations': [],
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'learning_rates': [],
            'game_stats': [],
            'timing': []
        }
    
    def update_history(self, iteration, policy_loss, value_loss, total_loss, learning_rate, 
                      game_stats, selfplay_time, training_time, total_time):
        """Update training history with metrics from current iteration"""
        self.training_history['iterations'].append(iteration)
        self.training_history['policy_losses'].append(policy_loss)
        self.training_history['value_losses'].append(value_loss)
        self.training_history['total_losses'].append(total_loss)
        self.training_history['learning_rates'].append(learning_rate)
        self.training_history['game_stats'].append(game_stats)
        self.training_history['timing'].append({
            'selfplay_time': selfplay_time,
            'training_time': training_time,
            'total_time': total_time
        })
    
    def save_training_history(self, args):
        """Save training history to JSON file"""
        history_filename = (
            f"training_history_"
            f"iter{args['num_iterations']}_"
            f"games{args['num_self_play_iterations']}_"
            f"mcts{args['num_searches']}_"
            f"{self.timestamp}.json"
        )
        history_file = os.path.join(self.log_dir, history_filename)
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        if self.logger:
            self.logger.info(f"Training history saved to {history_file}")
    
    def estimate_time_remaining(self, current_iter, total_iterations):
        """Estimate remaining training time"""
        if current_iter == 0:
            return "Unknown"
        
        elapsed = time.time() - self.start_time
        remaining_iters = total_iterations - current_iter
        time_per_iter = elapsed / current_iter
        remaining_seconds = remaining_iters * time_per_iter
        
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    
    def log_iteration_summary(self, iteration, iteration_stats):
        """Log clean iteration summary with time estimates"""
        if not self.logger:
            return
            
        balance_status = ""
        if iteration_stats['p1_win_rate'] > 0.7:
            balance_status = f" [P1 bias {iteration_stats['p1_win_rate']:.1%}]"
        elif iteration_stats['p_neg1_win_rate'] > 0.7:
            balance_status = f" [P-1 bias {iteration_stats['p_neg1_win_rate']:.1%}]"
        else:
            balance_status = " [Balanced]"
        
        time_remaining = self.estimate_time_remaining(iteration + 1, iteration_stats.get('total_iterations', 1))
        
        self.logger.info(
            f"Iter {iteration:2d}: "
            f"Games {iteration_stats['total_games']} | "
            f"P1:{iteration_stats['p1_wins']} P-1:{iteration_stats['p_neg1_wins']} D:{iteration_stats['draws']} | "
            f"Avg moves {iteration_stats['avg_game_length']:.1f} | "
            f"Policy {iteration_stats['policy_loss']:.3f} Value {iteration_stats['value_loss']:.3f} | "
            f"LR {iteration_stats['learning_rate']:.4f} | "
            f"Time {iteration_stats['total_time']:.1f}s | "
            f"ETA {time_remaining}"
            f"{balance_status}"
        )