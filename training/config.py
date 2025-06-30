# training/config.py

import json
import os
import sys


def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"   Loaded configuration from: {config_path}")
        print(f"   Description: {config.get('description', 'No description')}")
        print(f"   Estimated time: {config.get('estimated_time', 'Unknown')}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found")
        print("Available configurations:")
        for file in os.listdir('.'):
            if file.endswith('.json'):
                print(f"  - {file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{config_path}': {e}")
        sys.exit(1)


def validate_config(config):
    """Validate configuration has required sections"""
    required_sections = ['game_config', 'model_config', 'training_args', 'optimizer_config']
    missing_sections = [section for section in required_sections if section not in config]
    
    if missing_sections:
        print(f"Error: Missing required sections in config: {missing_sections}")
        sys.exit(1)
    
    print("Configuration validation passed")


def print_config_summary(config):
    """Print a nicely formatted configuration summary"""
    print(f"\n{'='*80}")
    print(f"CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"Description: {config.get('description', 'No description')}")
    print(f"Estimated Time: {config.get('estimated_time', 'Unknown')}")
    print(f"Use Case: {config.get('use_case', 'Not specified')}")
    
    print(f"\nGAME CONFIGURATION:")
    game_config = config['game_config']
    print(f"  Board size: {game_config['row_count']}x{game_config['col_count']}")
    print(f"  Buffer count: {game_config['buffer_count']}")
    print(f"  Draw move limit: {game_config['draw_move_limit']}")
    print(f"  Repetition limit: {game_config['repetition_limit']}")
    print(f"  History timesteps: {game_config['history_timesteps']}")
    
    print(f"\nMODEL CONFIGURATION:")
    model_config = config['model_config']
    print(f"  ResNet blocks: {model_config['num_resBlocks']}")
    print(f"  Hidden units: {model_config['num_hidden']}")
    print(f"  Use gradient checkpointing: {model_config.get('use_checkpoint', False)}")
    
    print(f"\nTRAINING CONFIGURATION:")
    training_args = config['training_args']
    print(f"  Iterations: {training_args['num_iterations']}")
    print(f"  Self-play games per iteration: {training_args['num_self_play_iterations']}")
    print(f"  Training epochs: {training_args['num_epochs']}")
    print(f"  Batch size: {training_args['batch_size']}")
    print(f"  MCTS simulations: {training_args['num_searches']}")
    print(f"  UCB constant (C): {training_args['C']}")
    print(f"  Temperature: {training_args.get('initial_temperature', 1.0)} -> {training_args.get('final_temperature', 1.0)}")
    print(f"  Dirichlet noise: ε={training_args['dirichlet_epsilon']}, α={training_args['dirichlet_alpha']}")
    print(f"  LR decay: every {training_args['lr_decay_steps']} iterations by {training_args['lr_decay_factor']}x")
    
    print(f"\nOPTIMIZER CONFIGURATION:")
    optimizer_config = config['optimizer_config']
    print(f"  Learning rate: {optimizer_config['lr']}")
    print(f"  Weight decay (L2 reg): {optimizer_config['weight_decay']}")
    print(f"  Mixed precision: {config.get('use_mixed_precision', True)}")
    
    if 'notes' in config and config['notes']:
        print(f"\nNOTES:")
        for note in config['notes']:
            print(f"  • {note}")
    
    print(f"{'='*80}")