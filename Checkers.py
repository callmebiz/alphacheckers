# Checkers.py

import numpy as np
from collections import defaultdict


class Checkers:
    def __init__(self, row_count=8, col_count=8, buffer_count=2, draw_move_limit=50, 
                 repetition_limit=3, history_timesteps=4):
        self.row_count = row_count
        self.col_count = col_count
        self.buffer_count = buffer_count
        self.draw_move_limit = draw_move_limit
        self.repetition_limit = repetition_limit  # How many repetitions trigger a draw
        self.history_timesteps = history_timesteps  # Number of historical board states to track
        
        self.playable_positions = []
        self.move_map = {}  # Maps each playable position to its valid moves
        
        self.move_names = {
            (-1, -1): "Up-Left", (-1, 1): "Up-Right",
            (1, -1): "Down-Left", (1, 1): "Down-Right",
            (-2, -2): "Jump Up-Left", (-2, 2): "Jump Up-Right",
            (2, -2): "Jump Down-Left", (2, 2): "Jump Down-Right"
        }
        
        self.action_size = self.initialize_action_encoding()
        print(f"Total Action Size: {self.action_size}")
        print(f"History Timesteps: {self.history_timesteps}")
        print(f"Repetition Limit: {self.repetition_limit}")

    def initialize_action_encoding(self):
        """
        Determines the valid actions for each playable square and sets action_size.
        """
        action_size = 0

        for r in range(self.row_count):
            for c in range(self.col_count):
                if (r + c) % 2 == 0:
                    continue  # Skip non-playable squares
                
                valid_moves = []
                for (dr, dc) in self.move_names.keys():
                    new_r, new_c = r + dr, c + dc
                    
                    if 0 <= new_r < self.row_count and 0 <= new_c < self.col_count:
                        valid_moves.append((new_r, new_c, action_size, dr, dc))
                        action_size += 1

                self.playable_positions.append((r, c))
                self.move_map[(r, c)] = valid_moves

        return action_size
        
    def load_state(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()

        # Extract the player's turn from the first line
        player = int(lines[0].strip())

        # Extract board lines (ignore header and borders)
        board_data = []
        for line in lines[2:]:  # Skip player turn and first empty line
            parts = line.strip().split("|")
            if len(parts) > 1:  # Ensure there's a board row present
                row_values = parts[-1].strip().split()  # Extract numbers
                board_data.append(list(map(int, row_values)))

        # Convert list to NumPy array
        board = np.array(board_data, dtype=int)

        # Initialize the state dictionary
        state = {
            'board': board,
            'state_repetitions': defaultdict(int),  # Count occurrences of each position
            'no_progress_moves': 0,
            'jump_again': None,
            'board_timesteps': [board.copy()]  # Historical board positions for neural network
        }

        # Compute initial board hash and update position counts
        board_hash = state['board'].tobytes()
        state['state_repetitions'][board_hash] = 1

        return state, player
    
    def show_board(self, state, player=None):
        """Prints the checkers board with proper alignment for any board size."""
        board = state['board']        
        # Determine max width required for column numbers and board values
        max_col_width = len(str(self.col_count - 1))  # Width of largest column index
        max_piece_width = max(len(str(np.max(board))), len(str(np.min(board))))  # Piece width
        
        cell_width = max(max_col_width, max_piece_width) + 1  # Ensure spacing

        # Construct column numbers with proper spacing
        col_numbers = " " + " " * (cell_width + 2) + " ".join(f"{i:>{cell_width}}" for i in range(self.col_count))
        
        # Construct horizontal border
        border = " " * (cell_width + 1) + "-" * ((cell_width + 1) * self.col_count + 1)
        
        print(col_numbers)
        print(border)
        
        # Print board rows with aligned values
        for i, row in enumerate(board):
            row_str = f"{i:>{cell_width}} | " + " ".join(f"{val:>{cell_width}}" for val in row)
            print(row_str)
        
        print()  # Extra newline for readability
        
        # Show valid moves if player is given
        if player:
            print(f"Player {player}")
            valid_moves = self.get_valid_moves(state, player)
            self.show_valid_moves(valid_moves)
    
    def show_policy(self, policy, value, k):
        """
        Displays the top k actions with the highest probability from the given policy.
        Prints the starting position, destination, move type, and policy index.
        """
        move_offsets = [
            (-1, -1, "Up-Left"), (-1, 1, "Up-Right"),
            (1, -1, "Down-Left"), (1, 1, "Down-Right"),
            (-2, -2, "Jump Up-Left"), (-2, 2, "Jump Up-Right"),
            (2, -2, "Jump Down-Left"), (2, 2, "Jump Down-Right")
        ]

        # Get the top k action indices sorted by probability
        top_k_indices = np.argsort(policy)[-k:][::-1]  # Sort descending

        print(f"Value: {value:.3f}")
        print("Top Policy Moves")
        print("----------------")

        for idx in top_k_indices:
            position_index = idx // 8
            move_type = idx % 8

            from_row = position_index // self.col_count
            from_col = position_index % self.col_count

            dr, dc, move_name = move_offsets[move_type]
            to_row = from_row + dr
            to_col = from_col + dc

            print(f"({from_row}, {from_col}) -> ({to_row}, {to_col}) ({move_name}) [idx={idx}, prob={policy[idx]:.3f}]")

    def get_initial_state(self):
        """
        Returns the initial game state as a dictionary containing:
        - 'board': NumPy array representing the board.
        - 'state_repetitions': defaultdict to track position repetitions.
        - 'no_progress_moves': Counter for no-capture moves.
        - 'jump_again': Flag to enforce multi-jump moves.
        - 'board_timesteps': List of previous board states for neural network input.
        """
        board = np.zeros((self.row_count, self.col_count), dtype=int)
        player_rows = (self.row_count - self.buffer_count) // 2

        # Place pieces for Player -1 (Top)
        for r in range(player_rows):
            for c in range(self.col_count):
                if (r + c) % 2 == 1:
                    board[r, c] = -1  

        # Place pieces for Player 1 (Bottom)
        for r in range(self.row_count - player_rows, self.row_count):
            for c in range(self.col_count):
                if (r + c) % 2 == 1:
                    board[r, c] = 1  

        return {
            'board': board,
            'state_repetitions': defaultdict(int),
            'no_progress_moves': 0,
            'jump_again': None,
            'board_timesteps': [board.copy()]  # Initialize with current board
        }

    def get_valid_moves(self, state, player, enforce_piece=None):
        """
        Returns a 1D binary array indicating valid moves.
        Normal pieces move/jump in one direction, kings move/jump in all directions.
        If a jump move is possible, only jump moves are valid.
        """
        board = state['board']
        valid_moves = np.zeros(self.action_size, dtype=int)
        jump_moves_exist = False
        potential_jumps = []

        # Iterate through all playable positions
        for (r, c) in self.playable_positions:
            piece = board[r, c]
            if piece == 0 or (piece != player and piece != 2 * player):
                continue  # Skip empty squares or opponent pieces
            
            if state['jump_again'] and (r, c) != state['jump_again']:
                continue

            is_king = abs(piece) == 2  # Determine if the piece is a king
            if enforce_piece and (r, c) != enforce_piece:
                continue  # If enforcing a piece, skip other pieces

            for new_r, new_c, action_idx, dr, dc in self.move_map[(r, c)]:
                # Jump move (two steps)
                if abs(new_r - r) == 2:
                    mid_r, mid_c = (r + new_r) // 2, (c + new_c) // 2
                    if board[mid_r, mid_c] in {-player, -2 * player} and board[new_r, new_c] == 0:
                        if is_king or (player == 1 and dr == -2) or (player == -1 and dr == 2):
                            valid_moves[action_idx] = 1
                            jump_moves_exist = True
                            potential_jumps.append(action_idx)

                # Normal move (one step)
                elif board[new_r, new_c] == 0:
                    if not enforce_piece and (is_king or (player == 1 and dr == -1) or (player == -1 and dr == 1)):
                        valid_moves[action_idx] = 1

        # If a jump move exists, remove all normal moves
        if jump_moves_exist:
            valid_moves = np.zeros(self.action_size, dtype=int)
            for action_idx in potential_jumps:
                valid_moves[action_idx] = 1

        return valid_moves

    def get_next_state(self, state, action, player):
        """
        Returns the new game state after applying the given action.
        """
        new_state = {
            'board': state['board'].copy(),
            'state_repetitions': state['state_repetitions'].copy(),
            'no_progress_moves': state['no_progress_moves'],
            'jump_again': state['jump_again'],
            'board_timesteps': state['board_timesteps'].copy()
        }
        
        for (r, c), moves in self.move_map.items():
            for new_r, new_c, action_idx, _, _ in moves:
                if action == action_idx:
                    new_state['board'][new_r, new_c] = new_state['board'][r, c]
                    new_state['board'][r, c] = 0
                    
                    # Handle capture
                    capture_made = False
                    if abs(new_r - r) == 2:
                        mid_r, mid_c = (r + new_r) // 2, (c + new_c) // 2
                        new_state['board'][mid_r, mid_c] = 0
                        capture_made = True
                        
                        if self.get_valid_moves(new_state, player, enforce_piece=(new_r, new_c)).any():
                            new_state['jump_again'] = (new_r, new_c)
                        else:
                            new_state['jump_again'] = None
                    else:
                        new_state['jump_again'] = None
                    
                    # Handle promotion
                    promotion_made = False
                    if ((player == 1 and new_r == 0) or (player == -1 and new_r == self.row_count - 1)) and abs(new_state['board'][new_r, new_c]) == 1:
                        new_state['board'][new_r, new_c] *= 2
                        promotion_made = True
                    
                    # Reset move counter if capture or promotion
                    if capture_made or promotion_made:
                        new_state['no_progress_moves'] = 0
                    else:
                        new_state['no_progress_moves'] += 1
                    
                    # Update position counts
                    board_hash = new_state['board'].tobytes()
                    new_state['state_repetitions'][board_hash] += 1
                    
                    # Update board timesteps (only when turn actually ends, not during multi-jumps)
                    if new_state['jump_again'] is None:
                        new_state['board_timesteps'].append(new_state['board'].copy())
                        # Keep only the last history_timesteps boards
                        if len(new_state['board_timesteps']) > self.history_timesteps:
                            new_state['board_timesteps'] = new_state['board_timesteps'][-self.history_timesteps:]
                    
                    return new_state
                    
        return new_state  # still return if no action is matched
    
    def show_valid_moves(self, valid_moves):
        print("Valid Moves")
        print("------------")

        move_list = []
        for (r, c), moves in self.move_map.items():
            for new_r, new_c, action_idx, dr, dc in moves:
                if valid_moves[action_idx] == 1:
                    move_name = self.move_names[(dr, dc)]
                    move_list.append(f"{len(move_list)}: ({r}, {c}) -> ({new_r}, {new_c}) ({move_name}) [idx={action_idx}]")

        for move in move_list:
            print(move)
            
    def check_win(self, state, player):
        """Returns True if the given player has won, meaning the opponent has no pieces or no valid moves."""
        opponent = self.get_opponent(player)

        # Check if opponent has any pieces left
        opponent_pieces = np.where((state['board'] == opponent) | (state['board'] == 2 * opponent))
        if len(opponent_pieces[0]) == 0:
            return True  # Opponent has no pieces, player wins

        # Check if opponent has any valid moves left
        state_copy = state.copy()
        state_copy['jump_again'] = None
        valid_moves_opponent = self.get_valid_moves(state_copy, opponent)
        if not valid_moves_opponent.any():
            return True  # No valid moves, player wins by stalemate
        return False
    
    def check_draw(self, state):
        """Returns True if the game is a draw (repetition limit or no-progress limit reached)."""
        # Check repetition limit
        if any(count >= self.repetition_limit for count in state['state_repetitions'].values()):
            return True

        # Check no-progress move limit
        if state['no_progress_moves'] >= self.draw_move_limit:
            return True

        return False
    
    def get_value_and_terminated(self, state, player):
        """Determine if the game is over."""
        if self.check_win(state, player):
            return 1, True  # Player wins
        
        if self.check_win(state, -player):
            return -1, True  # Opponent wins

        if self.check_draw(state):
            return 0, True  # Draw

        return 0, False  # Game continues
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def get_encoded_state(self, state, player):
        """
        Enhanced state encoding with historical timesteps and additional features.
        
        Returns state encoding with:
        - Historical board positions (4 planes × history_timesteps)
        - Current position repetition count (1 plane)
        - No progress moves count (1 plane)
        - Player color (1 plane)
        
        Total planes: 4 * history_timesteps + 3
        """
        board = state['board']
        board_timesteps = state.get('board_timesteps', [board])
        
        # Ensure we have enough timesteps (pad with zeros if needed)
        timesteps_needed = self.history_timesteps
        while len(board_timesteps) < timesteps_needed:
            zero_board = np.zeros_like(board)
            board_timesteps = [zero_board] + board_timesteps
        
        # Take only the most recent timesteps
        recent_timesteps = board_timesteps[-timesteps_needed:]
        
        # Create timestep planes for each historical board
        timestep_planes = []
        for timestep_board in recent_timesteps:
            if player == 1:
                # Player 1's perspective
                planes = [
                    timestep_board == -2,  # Opponent kings
                    timestep_board == -1,  # Opponent normal pieces
                    timestep_board == 1,   # Player normal pieces  
                    timestep_board == 2    # Player kings
                ]
            else:  # player == -1
                # Player -1's perspective  
                planes = [
                    timestep_board == 2,   # Opponent kings
                    timestep_board == 1,   # Opponent normal pieces
                    timestep_board == -1,  # Player normal pieces
                    timestep_board == -2   # Player kings
                ]
            timestep_planes.extend(planes)
        
        # Additional feature planes
        # Current position repetition count (normalized)
        board_hash = board.tobytes()
        current_repetitions = state['state_repetitions'].get(board_hash, 0)
        repetition_plane = np.full((self.row_count, self.col_count), 
                                 min(current_repetitions / self.repetition_limit, 1.0), 
                                 dtype=np.float32)
        
        # No progress moves count (normalized)
        no_progress_plane = np.full((self.row_count, self.col_count), 
                                  min(state['no_progress_moves'] / self.draw_move_limit, 1.0), 
                                  dtype=np.float32)
        
        # Player color (1 for player 1, 0 for player -1)
        player_plane = np.full((self.row_count, self.col_count), 
                             1.0 if player == 1 else 0.0, 
                             dtype=np.float32)
        
        # Combine all planes
        all_planes = timestep_planes + [repetition_plane, no_progress_plane, player_plane]
        encoded_state = np.stack(all_planes).astype(np.float32)
        
        return encoded_state
    
    def get_num_channels(self):
        """Return the number of channels in the encoded state."""
        return 4 * self.history_timesteps + 3
    
    def decode_encoded_state(self, encoded_state, player):
        """
        Converts an encoded state back into the most recent board representation.
        """
        board = np.zeros((self.row_count, self.col_count), dtype=int)
        
        # The most recent board state is in the last 4 channels of timestep data
        recent_start = (self.history_timesteps - 1) * 4
        recent_planes = encoded_state[recent_start:recent_start + 4]
        
        if player == 1:
            board[recent_planes[0] == 1] = -2  # Opponent kings
            board[recent_planes[1] == 1] = -1  # Opponent normal pieces
            board[recent_planes[2] == 1] = 1   # Player normal pieces
            board[recent_planes[3] == 1] = 2   # Player kings
        else:  # player == -1
            board[recent_planes[0] == 1] = 2   # Opponent kings
            board[recent_planes[1] == 1] = 1   # Opponent normal pieces  
            board[recent_planes[2] == 1] = -1  # Player normal pieces
            board[recent_planes[3] == 1] = -2  # Player kings

        return board


if __name__ == "__main__":
    checkers = Checkers(
        row_count=8, 
        col_count=8, 
        buffer_count=2, 
        draw_move_limit=50,
        repetition_limit=3,
        history_timesteps=4
    )
    state = checkers.get_initial_state()
    player = 1

    while True:
        checkers.show_board(state)

        # Get valid moves for the current player
        valid_moves = checkers.get_valid_moves(state, player)

        # Extract valid move indices
        move_mapping = [idx for idx, is_valid in enumerate(valid_moves) if is_valid]

        # Print valid moves
        checkers.show_valid_moves(valid_moves)

        # Get user input
        quit = False
        while True:
            action = input(f"Player {player}, select a move (0-{len(move_mapping) - 1}) or type 'q' to quit: ").strip()

            if action.lower() == "q":
                quit = True
                break

            try:
                action = int(action)
                if 0 <= action < len(move_mapping):
                    action = move_mapping[action]
                    break
            except ValueError:
                pass
                
            print("Invalid input! Enter a number corresponding to a valid move or 'q' to quit.")

        if quit:
            print("\nGame ended manually.")
            break

        # Apply the selected move
        state = checkers.get_next_state(state, action, player)

        # Check if the game has ended
        value, is_terminal = checkers.get_value_and_terminated(state, player)
        if is_terminal:
            checkers.show_board(state)

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
                    print("Reason: Stalemate.")
            break

        # Switch players if not multi-jumping
        if state['jump_again'] is None:
            player = checkers.get_opponent(player)