# train.py

import torch
import argparse
import os

from core.game import Checkers
from core.state_encoder import StateEncoder
from core.model import ResNet
from core.mcts import MCTS
from training.config import load_config, validate_config, print_config_summary
from training.trainer import Trainer


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Enhanced AlphaZero Checkers Training with All Optimizations')
    parser.add_argument('config', help='Path to JSON configuration file (e.g., debug.json, dev.json, full_scale.json)')
    parser.add_argument('--model-dir', default='saved_models', help='Directory to save models (default: saved_models)')
    parser.add_argument('--log-dir', default='logs', help='Directory to save logs (default: logs)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to use (default: auto)')
    parser.add_argument('--resume', help='Path to checkpoint to resume from (e.g., saved_models/checkpoint_latest.pt)')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    validate_config(config)
    
    # Print configuration summary
    print_config_summary(config)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Initialize game with configuration
    game_config = config['game_config']
    game = Checkers(
        row_count=game_config['row_count'],
        col_count=game_config['col_count'],
        buffer_count=game_config['buffer_count'],
        draw_move_limit=game_config['draw_move_limit'],
        repetition_limit=game_config['repetition_limit'],
        history_timesteps=game_config['history_timesteps']
    )
    
    # Initialize state encoder
    state_encoder = StateEncoder(game)
    
    # Initialize neural network with configuration
    model_config = config['model_config']
    model = ResNet(
        input_channels=state_encoder.get_num_channels(),
        action_size=game.action_size,
        board_size=(game.row_count, game.col_count),
        num_resBlocks=model_config['num_resBlocks'], 
        num_hidden=model_config['num_hidden'], 
        device=device,
        use_checkpoint=model_config.get('use_checkpoint', False)
    )
    
    # Initialize optimizer with configuration
    optimizer_config = config['optimizer_config']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config['weight_decay']
    )
    
    # Get training arguments from configuration
    training_args = config['training_args']
    
    # Initialize MCTS
    mcts = MCTS(game, training_args, model, state_encoder)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Print final summary
    print(f"\nSTARTING TRAINING:")
    print(f"   Neural network input channels: {state_encoder.get_num_channels()}")
    print(f"   Action space size: {game.action_size}")
    print(f"   Model parameters: ~{sum(p.numel() for p in model.parameters()):,}")
    print(f"   Configuration: {args.config}")
    print(f"   Model directory: {args.model_dir}")
    print(f"   Log directory: {args.log_dir}")
    if args.resume:
        print(f"   Resume from: {args.resume}")
    print()
    
    # Initialize and start training
    initial_state = None
    use_mixed_precision = not args.no_mixed_precision and torch.cuda.is_available()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        game=game,
        mcts=mcts,
        state_encoder=state_encoder,
        args=training_args,
        initial_state=initial_state,
        log_dir=args.log_dir,
        use_mixed_precision=use_mixed_precision
    )
    
    trainer.train(model_dir=args.model_dir, resume_from=args.resume)


if __name__ == "__main__":
    main()