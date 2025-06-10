import numpy as np
from collections import defaultdict
import hashlib


class Checkers:
    def __init__(self, row_count=8, col_count=8, buffer_count=2, draw_move_limit=50):
        self.row_count = row_count
        self.col_count = col_count
        self.action_size = self.row_count * self.col_count * 8  # Encoding 8 moves per position
        self.buffer_count = buffer_count
        self.draw_move_limit = draw_move_limit  # Moves limit for no capture/promotion draw
        self.board = self.get_initial_state()
        
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

    
    def print_board(self, state):
        """Prints the checkers board with proper alignment for any board size."""
        state = state['board']        
        # Determine max width required for column numbers and board values
        max_col_width = len(str(self.col_count - 1))  # Width of largest column index
        max_piece_width = max(len(str(np.max(state))), len(str(np.min(state))))  # Piece width
        
        cell_width = max(max_col_width, max_piece_width) + 1  # Ensure spacing

        # Construct column numbers with proper spacing
        col_numbers = " " + " " * (cell_width + 2) + " ".join(f"{i:>{cell_width}}" for i in range(self.col_count))
        
        # Construct horizontal border
        border = " " * (cell_width + 1) + "-" * ((cell_width + 1) * self.col_count + 1)
        
        print(col_numbers)
        print(border)
        
        # Print board rows with aligned values
        for i, row in enumerate(state):
            row_str = f"{i:>{cell_width}} | " + " ".join(f"{val:>{cell_width}}" for val in row)
            print(row_str)
        
        print()  # Extra newline for readability
    
    def reset_game(self):
        """Resets the game state for a new match."""
        self.board = self.get_initial_state()
        self.board_history.clear()
        self.no_progress_moves = 0

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
    
    def print_valid_moves(self, valid_moves):
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
        elif self.check_win(state, self.get_opponent(player)):
            return -1, True  # Opponent wins

        if self.check_draw(state):
            return 0, True  # Draw

        return 0, False  # Game is still ongoing
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    
class Node:
    def __init__(self, game, args, state, player, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.player = player
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = []
        self.expandable_moves = game.get_valid_moves(state, player)
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        return best_child
    
    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value * self.args['C'] * np.sqrt(np.log(self.visit_count) / child.visit_count)
        
    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == True)[0])
        self.expandable_moves[action] = False
        
        child_state = self.state.copy()
        child_state = self.game.get_next_state(self.state, action, self.player)
        
        player = self.player * -1  # Switch players unless another jump is required
        if child_state['jump_again']:
            player = self.player
            
        child = Node(self.game, self.args, child_state, player, self, action)
        self.children.append(child)
        return child
    
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.player)
        value = self.game.get_opponent_value(value)
        
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = self.player
        while True:
            try:
                vaid_moves = self.game.get_valid_moves(rollout_state, rollout_player)
            except:
                pass
            action = np.random.choice(np.where(vaid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, rollout_player)
            if is_terminal:
                if rollout_player != self.player:
                    value = self.game.get_opponent_value(value)
                return value
            if not rollout_state['jump_again']:  # Keep same player if another jump move is required
                rollout_player = self.game.get_opponent(rollout_player)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

from tqdm import tqdm
class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args
    
    def search(self, state, player):
        root = Node(self.game, self.args, state, player)
        
        for _ in tqdm(range(self.args['num_searches'])):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.player)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                node = node.expand()
                value = node.simulate()
                
            node.backpropagate(value)
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
                
    
    
if __name__ == "__main__":
    # np.random.seed(1)
    state_path = "state_multiJump2.txt"
    state_path = None
    checkers = Checkers(row_count=8, col_count=8, buffer_count=2, draw_move_limit=50)

    if state_path:
        state, player = checkers.load_state(state_path)
    else:
        state = checkers.get_initial_state()
        player = 1

    args = {
        'C': 2.41,
        'num_searches': 100000
    }
    mcts = MCTS(checkers, args)

    while True:
        checkers.print_board(state)

        # Get valid moves for the current player
        valid_moves = checkers.get_valid_moves(state, player)
        move_mapping = [idx for idx, is_valid in enumerate(valid_moves) if is_valid]

        # Print valid moves
        print(f"\nPlayer {player}'s Turn")
        checkers.print_valid_moves(valid_moves)

        if player == 1:
            # Player's turn (manual input)
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
        else:
            # MCTS turn (AI's move)
            mcts_probs = mcts.search(state, player)
            action = np.argmax(mcts_probs)
            
            # Print action probabilities
            print("\nMCTS Action Probabilities:")
            for idx, move_idx in enumerate(move_mapping):
                print(f"  {idx}: ({move_idx}) -> {mcts_probs[move_idx]:.4f}")

            print(f"\nMCTS selected move index: {action}")

        # Apply the selected move
        state = checkers.get_next_state(state, action, player)

        # Check if the game has ended
        value, is_terminal = checkers.get_value_and_terminated(state, player)
        if is_terminal:
            checkers.print_board(state)

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

