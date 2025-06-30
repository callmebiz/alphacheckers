# play.py

import numpy as np
import torch
import sys
import os

from Checkers import Checkers
from ResNet import ResNet


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint file"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None, None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract game configuration
        game_config = checkpoint.get('game_config', {})
        
        # Initialize game with checkpoint config
        game = Checkers(
            row_count=game_config.get('row_count', 8),
            col_count=game_config.get('col_count', 8),
            buffer_count=game_config.get('buffer_count', 2),
            draw_move_limit=game_config.get('draw_move_limit', 50),
            repetition_limit=game_config.get('repetition_limit', 3),
            history_timesteps=game_config.get('history_timesteps', 4)
        )
        
        # Extract model configuration
        model_config = checkpoint['model_config']
        
        num_resblocks = model_config['num_resBlocks']
        num_hidden = model_config['num_hidden']
        use_checkpoint = model_config.get('use_checkpoint', False)
        
        print(f"Model config: {num_resblocks} ResBlocks, {num_hidden} hidden units")
        
        # Initialize model
        model = ResNet(game, num_resblocks, num_hidden, device, use_checkpoint=use_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        iteration = checkpoint.get('iteration', 'unknown')
        print(f"Loaded checkpoint from iteration {iteration}")
        print(f"Model expects {game.get_num_channels()} input channels")
        
        return model, game
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None


def load_model(model_path, game, device):
    """Load a saved model (legacy format)"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        model = ResNet(game, 8, 64, device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Loaded model from {model_path}")
        print(f"Model expects {game.get_num_channels()} input channels")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"This could be due to mismatched parameters (history_timesteps, etc.)")
        return None


def get_ai_move(model, game, state, player, device, temperature=0.1):
    """Get AI move using model output probabilities with temperature control"""
    with torch.no_grad():
        encoded_state = game.get_encoded_state(state, player)
        policy, value = model(torch.tensor(encoded_state, device=device).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        value = value.item()
    
    # Mask invalid moves
    valid_moves = game.get_valid_moves(state, player)
    policy *= valid_moves
    
    if np.sum(policy) == 0:
        return None, None, value
    
    # Apply temperature
    if temperature == 0:
        # Greedy selection
        action_probs = np.zeros_like(policy)
        action_probs[np.argmax(policy)] = 1.0
    else:
        # Temperature scaling
        policy = policy.astype(np.float64)
        if temperature != 1.0:
            policy = np.power(policy, 1.0 / temperature)
        policy /= np.sum(policy)  # Renormalize
        action_probs = policy
    
    return action_probs, valid_moves, value


def show_ai_move_probs(game, policy, valid_moves, value, player, temperature):
    """Show AI move probabilities"""
    print(f"\nAI Move Probabilities (Player {player}) [Value: {value:.3f}, Temp: {temperature:.2f}]:")
    print("-" * 75)
    
    # Get valid actions with probabilities
    valid_actions = []
    for action_idx, (valid, prob) in enumerate(zip(valid_moves, policy)):
        if valid and prob > 0:
            valid_actions.append((action_idx, prob))
    
    # Sort by probability (highest first)
    valid_actions.sort(key=lambda x: x[1], reverse=True)
    
    # Show moves in probability order
    for display_idx, (action_idx, prob) in enumerate(valid_actions):
        # Find move details
        for (r, c), moves in game.move_map.items():
            for new_r, new_c, act_idx, dr, dc in moves:
                if act_idx == action_idx:
                    move_name = game.move_names[(dr, dc)]
                    print(f"{display_idx}: ({r},{c}) -> ({new_r},{new_c}) {move_name:>12} [prob={prob:.3f}] [idx={action_idx}]")
                    break


def show_game_info(checkers, state):
    """Show current game information"""
    print(f"Position repetitions: {max(state['state_repetitions'].values()) if state['state_repetitions'] else 0}")
    print(f"No-progress moves: {state['no_progress_moves']}/{checkers.draw_move_limit}")
    print(f"Board timesteps stored: {len(state['board_timesteps'])}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python play.py <model_path_or_checkpoint> [mode] [history_timesteps] [repetition_limit] [temperature]")
        print("  mode: 'pvp' (default), 'pva', 'avp', 'ava'")
        print("  p = human player, a = AI")
        print("  history_timesteps: number of historical board states (default: 4)")
        print("  repetition_limit: position repetitions before draw (default: 3)")
        print("  temperature: AI temperature for move selection (default: 0.1, 0=greedy)")
        return
    
    model_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'pvp'
    history_timesteps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    repetition_limit = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    ai_temperature = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    
    if mode not in ['pvp', 'pva', 'avp', 'ava']:
        print("Invalid mode. Use 'pvp', 'pva', 'avp', or 'ava'")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to load as checkpoint first, then as regular model
    model = None
    checkers = None
    
    if model_path.endswith('.pt') and 'checkpoint' in model_path:
        print("Loading from checkpoint...")
        model, checkers = load_model_from_checkpoint(model_path, device)
    else:
        print("Loading as regular model...")
        # Initialize game with provided parameters
        checkers = Checkers(
            row_count=8, 
            col_count=8, 
            buffer_count=2, 
            draw_move_limit=50,
            repetition_limit=repetition_limit,
            history_timesteps=history_timesteps
        )
        model = load_model(model_path, checkers, device)
    
    if model is None or checkers is None:
        print("Failed to load model. Exiting.")
        return
    
    # Set up players
    player1_ai = mode in ['avp', 'ava']
    player2_ai = mode in ['pva', 'ava']
    
    print(f"\nGame Configuration:")
    print(f"  Mode: {mode}")
    print(f"  Player 1: {'AI' if player1_ai else 'Human'}")
    print(f"  Player -1: {'AI' if player2_ai else 'Human'}")
    print(f"  History timesteps: {checkers.history_timesteps}")
    print(f"  Repetition limit: {checkers.repetition_limit}")
    print(f"  Draw move limit: {checkers.draw_move_limit}")
    print(f"  Neural network channels: {checkers.get_num_channels()}")
    print(f"  AI temperature: {ai_temperature}")
    print("=" * 60)
    
    # Game loop
    state = checkers.get_initial_state()
    player = 1
    move_number = 0
    
    while True:
        move_number += 1
        print(f"\n--- Move {move_number} ---")
        
        checkers.show_board(state)
        show_game_info(checkers, state)
        
        # Check for game end
        value, is_terminal = checkers.get_value_and_terminated(state, player)
        if is_terminal:
            print(f"\nGame Over!")
            if value == 1:
                print(f"Player {player} won!")
            elif value == -1:
                print(f"Player {-player} won!")
            else:
                print("Game ended in a draw.")
                
                # Identify the draw reason
                if state['no_progress_moves'] >= checkers.draw_move_limit:
                    print("Reason: No-progress move limit reached.")
                elif any(count >= checkers.repetition_limit for count in state['state_repetitions'].values()):
                    print("Reason: Position repetition limit reached.")
                else:
                    print("Reason: Stalemate (no valid moves).")
            break
        
        # Determine if current player is AI
        is_ai = (player == 1 and player1_ai) or (player == -1 and player2_ai)
        
        if is_ai:
            # AI move with temperature control
            policy, valid_moves, ai_value = get_ai_move(model, checkers, state, player, device, ai_temperature)
            
            if policy is None:
                print(f"AI Player {player} has no valid moves!")
                break
            
            # Show AI probabilities
            show_ai_move_probs(checkers, policy, valid_moves, ai_value, player, ai_temperature)
            
            # Sample from probability distribution
            if ai_temperature == 0:
                # Greedy selection
                action = np.argmax(policy)
            else:
                # Stochastic selection
                action = np.random.choice(len(policy), p=policy)
            print(f"AI selects action {action}")
            
        else:
            # Human move
            valid_moves = checkers.get_valid_moves(state, player)
            move_mapping = [idx for idx, is_valid in enumerate(valid_moves) if is_valid]
            
            if not move_mapping:
                print(f"Player {player} has no valid moves!")
                break
            
            checkers.show_valid_moves(valid_moves)
            
            # Get human input
            quit = False
            while True:
                prompt = f"Player {player}, select a move (0-{len(move_mapping) - 1}) or 'q' to quit: "
                action = input(prompt).strip()

                if action.lower() == "q":
                    quit = True
                    break

                try:
                    action = int(action)
                    if 0 <= action < len(move_mapping):
                        action = move_mapping[action]  # Map to actual action index
                        break
                except ValueError:
                    pass
                    
                print("Invalid input! Enter a number corresponding to a valid move or 'q' to quit.")

            if quit:
                print("\nGame ended manually.")
                break
        
        # Apply the selected move
        state = checkers.get_next_state(state, action, player)

        # Switch players if not multi-jumping
        if state['jump_again'] is None:
            player = checkers.get_opponent(player)
        else:
            player_type = "AI" if is_ai else "Human"
            print(f"Multi-jump! {player_type} Player {player} continues...")
        
        # Pause for AI vs AI games
        if mode == 'ava':
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()