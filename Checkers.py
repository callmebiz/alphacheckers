# Checkers.py - FIXED VERSION

import numpy as np
from collections import defaultdict


class Checkers:
    def __init__(self, row_count=8, col_count=8, buffer_count=2, draw_move_limit=50):
        self.row_count = row_count
        self.col_count = col_count
        self.buffer_count = buffer_count
        self.draw_move_limit = draw_move_limit
        
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
            'board_history': defaultdict(int),
            'no_progress_moves': 0,
            'jump_again': None
        }

        # Compute initial board hash and update board history
        board_hash = state['board'].tobytes()
        state['board_history'][board_hash] = 1

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
        - 'board_history': defaultdict to track previous states.
        - 'no_progress_moves': Counter for no-capture moves.
        - 'jump_again': Flag to enforce multi-jump moves.
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
            'board_history': defaultdict(int),
            'no_progress_moves': 0,
            'jump_again': None
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
            'board_history': state['board_history'].copy(),
            'no_progress_moves': state['no_progress_moves'],
            'jump_again': state['jump_again']
        }
        
        for (r, c), moves in self.move_map.items():
            for new_r, new_c, action_idx, _, _ in moves:
                if action == action_idx:
                    new_state['board'][new_r, new_c] = new_state['board'][r, c]
                    new_state['board'][r, c] = 0
                    
                    # Handle capture
                    if abs(new_r - r) == 2:
                        mid_r, mid_c = (r + new_r) // 2, (c + new_c) // 2
                        new_state['board'][mid_r, mid_c] = 0
                        
                        if self.get_valid_moves(new_state, player, enforce_piece=(new_r, new_c)).any():
                            new_state['jump_again'] = (new_r, new_c)
                        else:
                            new_state['jump_again'] = None
                    else:
                        new_state['jump_again'] = None
                    
                    # Handle promotion
                    if ((player == 1 and new_r == 0) or (player == -1 and new_r == self.row_count - 1)) and abs(new_state['board'][new_r, new_c]) == 1:
                        new_state['board'][new_r, new_c] *= 2
                    
                    # Reset move counter if capture or promotion
                    if new_state['jump_again'] or new_state['board'][new_r, new_c] == 2 * player:
                        new_state['no_progress_moves'] = 0
                    else:
                        new_state['no_progress_moves'] += 1
                    
                    # Update board history
                    board_hash = new_state['board'].tobytes()
                    new_state['board_history'][board_hash] += 1
                    
                    return new_state
                    
        return new_state  # still return if not action is matched
    
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
        opponent = self.get_opponent(player)  # Opponent's piece value

        # Check if opponent has any pieces left
        opponent_pieces = np.where((state['board'] == opponent) | (state['board'] == 2 * opponent))  # Normal and king pieces
        if len(opponent_pieces[0]) == 0:
            return True  # Opponent has no pieces, player wins

        # Check if opponent has any valid moves left
        state_copy = state.copy()
        state_copy['jump_again'] = None
        valid_moves_opponent = self.get_valid_moves(state_copy, opponent)
        if not valid_moves_opponent.any():
            return True  # No valid moves, player wins by opponent's forced pass
        return False  # Game is still ongoing
    
    def check_draw(self, state):
        """Returns True if the game is a draw (threefold repetition or move limit without capture/promotion)."""
        # Check threefold repetition rule (draw due to repeated board states)
        if any(count >= 3 for count in state['board_history'].values()):
            return True

        # Check if move limit without capture/promotion is reached
        if state['no_progress_moves'] >= self.draw_move_limit:
            return True  # Draw due to move limit

        return False  # Game is still ongoing
    
    def get_value_and_terminated(self, state, player):
        """Determine if the game is over and log results if tracking stats."""
        if self.check_win(state, player):
            return 1, True  # Player wins
        
        if self.check_win(state, -player):
            return -1, True  # Opponent wins

        if self.check_draw(state):
            return 0, True  # Draw

        return 0, False  # Game is still ongoing
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def get_encoded_state(self, state, player):
        """
        FIXED: Consistent state encoding regardless of player
        """
        board = state['board']
        
        if player == 1:
            # Player 1's perspective: their pieces are positive in channels 2-3
            encoded_state = np.stack([
                board == -2,  # Opponent kings
                board == -1,  # Opponent normal pieces
                board == 1,   # Player normal pieces  
                board == 2    # Player kings
            ]).astype(np.float32)
        else:  # player == -1
            # Player -1's perspective: their pieces become positive in channels 2-3
            encoded_state = np.stack([
                board == 2,   # Opponent kings (was player 1's)
                board == 1,   # Opponent normal pieces (was player 1's)
                board == -1,  # Player normal pieces (now positive)
                board == -2   # Player kings (now positive)
            ]).astype(np.float32)
        
        return encoded_state
    
    def decode_encoded_state(self, encoded_state, player):
        """
        Converts an encoded state back into the board representation.

        Args:
        - encoded_state: The encoded board state as a NumPy array.
        - player: The player perspective used in encoding.

        Returns:
        - A NumPy array representing the board.
        """
        board = np.zeros((self.row_count, self.col_count), dtype=int)
        
        if player == 1:
            board[encoded_state[0] == 1] = -2  # Opponent kings
            board[encoded_state[1] == 1] = -1  # Opponent normal pieces
            board[encoded_state[2] == 1] = 1   # Player normal pieces
            board[encoded_state[3] == 1] = 2   # Player kings
        else:  # player == -1
            board[encoded_state[0] == 1] = 2   # Opponent kings
            board[encoded_state[1] == 1] = 1   # Opponent normal pieces  
            board[encoded_state[2] == 1] = -1  # Player normal pieces
            board[encoded_state[3] == 1] = -2  # Player kings

        return board


if __name__ == "__main__":
    checkers = Checkers(row_count=8, col_count=8, buffer_count=2, draw_move_limit=50)
    state = checkers.get_initial_state()
    player = 1
    # state, player = checkers.load_state("state_multiJump3.txt")

    while True:
        checkers.show_board(state)  # Print board with row/col indices

        # Get valid moves for the current player
        valid_moves = checkers.get_valid_moves(state, player)

        # Extract valid move indices (mapping user-friendly index -> actual move index)
        move_mapping = [idx for idx, is_valid in enumerate(valid_moves) if is_valid]

        # Print valid moves
        checkers.show_valid_moves(valid_moves)

        # Keep asking until a valid move is chosen
        quit = False
        while True:
            action = input(f"Player {player}, select a move (0-{len(move_mapping) - 1}) or type 'q' to quit: ").strip()

            if action.lower() == "q":
                quit = True
                break

            try:
                action = int(action)
                if 0 <= action < len(move_mapping):  # Ensure the input is within range
                    action = move_mapping[action]  # Map input to actual move index
                    break
            except ValueError:
                pass  # Ignore invalid input
                
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
                    print("Reason: 50-move rule triggered.")
                elif any(count >= 3 for count in state['board_history'].values()):
                    print("Reason: Threefold repetition.")
                else:
                    print("Reason: Stalemate.")
            break

        # If the last move didn't require another jump, switch players
        if state['jump_again'] is None:
            player = checkers.get_opponent(player)