import numpy as np
from collections import defaultdict
import hashlib


class Checkers:
    def __init__(self, row_count=8, col_count=8, buffer_count=2, draw_move_limit=50):
        self.row_count = row_count
        self.col_count = col_count
        self.buffer_count = buffer_count
        self.draw_move_limit = draw_move_limit
        
        self.playable_positions = []
        self.move_map = {}  # Maps each playable position to its valid moves
        self.action_size = 0
        
        self.initialize_action_encoding()
        
    def initialize_action_encoding(self):
        """
        Determines the valid actions for each playable square and sets action_size.
        """
        move_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)]
        action_index = 0

        for r in range(self.row_count):
            for c in range(self.col_count):
                if (r + c) % 2 == 0:
                    continue  # Skip non-playable squares
                
                valid_moves = []
                for move in move_offsets:
                    dr, dc = move
                    new_r, new_c = r + dr, c + dc
                    
                    if 0 <= new_r < self.row_count and 0 <= new_c < self.col_count:
                        valid_moves.append((new_r, new_c, action_index))
                        action_index += 1

                self.playable_positions.append((r, c))
                self.move_map[(r, c)] = valid_moves

        self.action_size = action_index
        print(f"Total Action Size: {self.action_size}")
        
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
        board_hash = int(hashlib.md5(state['board'].tobytes()).hexdigest(), 16)
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
        Returns a boolean array encoding valid moves for each piece.
        """
        board = state['board']
        valid_moves = np.zeros(self.row_count * self.col_count * 8)
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        jump_moves_exist = False

        for row in range(self.row_count):
            for col in range(self.col_count):
                piece = board[row, col]
                if piece == 0 or (piece != player and piece != 2 * player):
                    continue
                
                # Only check enforce_pice if given (for multi-jump moves)
                if enforce_piece and (row, col) != enforce_piece:
                    continue

                index = (row * self.col_count + col) * 8
                is_king = abs(piece) == 2
                valid_directions = [is_king or player == 1, is_king or player == 1,
                                    is_king or player == -1, is_king or player == -1]

                for move_idx, (dr, dc) in enumerate(directions):
                    if not valid_directions[move_idx]:
                        continue

                    new_row, new_col = row + dr, col + dc
                    jump_row, jump_col = row + 2 * dr, col + 2 * dc
                    move_slot, jump_slot = index + move_idx, index + move_idx + 4

                    if 0 <= new_row < self.row_count and 0 <= new_col < self.col_count:
                        if board[new_row, new_col] == 0 and not enforce_piece:
                            valid_moves[move_slot] = 1
                        elif (0 <= jump_row < self.row_count and 0 <= jump_col < self.col_count and
                              board[new_row, new_col] in {-player, -2 * player} and
                              board[jump_row, jump_col] == 0):
                            valid_moves[jump_slot] = 1
                            jump_moves_exist = True

        if jump_moves_exist:
            for i in range(len(valid_moves)):
                if i % 8 < 4:  # Non-jump moves are in the first 4 slots of each position
                    valid_moves[i] = 0

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
        
        ## TODO: r, c, nr, nr = self.action_to_move(action)
        
        position_index = action // 8
        move_type = action % 8
        row, col = position_index // self.col_count, position_index % self.col_count
        move_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)]
        dr, dc = move_offsets[move_type]
        new_row, new_col = row + dr, col + dc

        new_state['board'][new_row, new_col] = new_state['board'][row, col]
        new_state['board'][row, col] = 0

        capture, promotion = False, False
        if move_type >= 4:
            mid_row, mid_col = (row + new_row) // 2, (col + new_col) // 2
            new_state['board'][mid_row, mid_col] = 0
            capture = True

        if capture and self.get_valid_moves(new_state, player, enforce_piece=(new_row, new_col)).any():
            new_state['jump_again'] = (new_row, new_col)
        else:
            new_state['jump_again'] = None

        if ((player == 1 and new_row == 0) or (player == -1 and new_row == self.row_count - 1)) \
            and abs(new_state['board'][new_row, new_col]) == 1:
            new_state['board'][new_row, new_col] *= 2
            promotion = True

        if capture or promotion:
            new_state['no_progress_moves'] = 0
        else:
            new_state['no_progress_moves'] += 1

        board_hash = int(hashlib.md5(new_state['board'].tobytes()).hexdigest(), 16)
        new_state['board_history'][board_hash] += 1
        
        return new_state
    
    def show_valid_moves(self, valid_moves):
        move_offsets = [
            (-1, -1, "Up-Left"), (-1, 1, "Up-Right"),
            (1, -1, "Down-Left"), (1, 1, "Down-Right"),
            (-2, -2, "Jump Up-Left"), (-2, 2, "Jump Up-Right"),
            (2, -2, "Jump Down-Left"), (2, 2, "Jump Down-Right")
        ]

        print("Valid Moves")
        print("------------")

        move_list = []
        for idx, val in enumerate(valid_moves):
            if val == 1:
                position_index = idx // 8
                move_type = idx % 8

                from_row = position_index // self.col_count
                from_col = position_index % self.col_count

                dr, dc, move_name = move_offsets[move_type]
                to_row = from_row + dr
                to_col = from_col + dc

                move_list.append(f"{len(move_list)}: ({from_row}, {from_col}) -> ({to_row}, {to_col}) ({move_name}) \t[idx={idx}]")

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
        valid_moves_opponent = self.get_valid_moves(state, opponent)
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

        if self.check_draw(state):
            return 0, True  # Draw

        return 0, False  # Game is still ongoing
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def get_encoded_state(self, state, player, ver=1):
        board = state['board']
        if ver == 1:
            encoded_state = np.stack(
                (board == -2, board == -1, board == 1, board == 2)  # encode each piece type
            ).astype(np.float32) * player  # different encoding for different players due do different valid actions per player
        else:
            encoded_state = board.astype(np.float32) * player
        return encoded_state


if __name__ == "__main__":
    checkers = Checkers(row_count=8, col_count=8, buffer_count=2, draw_move_limit=50)
    state = checkers.get_initial_state()
    player = 1
    state, player = checkers.load_state("state_multiJump3.txt")

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
        if not state['jump_again']:
            player = checkers.get_opponent(player)
