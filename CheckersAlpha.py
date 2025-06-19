# CheckersAlpha.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import json
import hashlib

from Checkers2 import Checkers


# torch.manual_seed(0)

class Node:
    def __init__(self, game, args, state, player, parent=None, action_taken=None, prior=0, visit_count=0, depth=0):
        self.game = game
        self.args = args
        self.state = state
        self.player = player
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.visit_count = visit_count
        self.depth = depth
        
        self.children = []
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
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
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * np.sqrt(self.visit_count / (child.visit_count + 1)) * child.prior
        
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(self.state, action, self.player)
                
                player = self.game.get_opponent(self.player)  # Switch players unless another jump is required
                if child_state['jump_again'] is not None:
                    player = self.player
                    
                child = Node(self.game, self.args, child_state, player, self, action, prob, depth=self.depth + 1)
                self.children.append(child)
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

    def inspect(self, file_path="tree_inspect.json", max_depth=None):
        """
        Saves the tree structure in JSON format for better visualization.
        
        Args:
            file_path (str): Path to save the JSON file.
            max_depth (int): If provided, limits the depth of inspection.
        """
        tree_data = self._get_tree_data(max_depth)
        
        with open(file_path, "w") as file:
            json.dump(tree_data, file, indent=4)

    def _get_tree_data(self, max_depth=None, level=0):
        """
        Recursively gathers tree structure data as a dictionary.
        Converts numpy.float32 values to Python floats to ensure JSON serialization.
        
        Returns:
            dict: Tree data formatted as JSON-compatible structure.
        """
        node_data = {
            "action": int(self.action_taken) if self.action_taken is not None else None,
            "player": int(self.player),
            "visit_count": int(self.visit_count),
            "value_sum": float(self.value_sum),  # Convert to Python float
            "prior": float(self.prior),  # Convert to Python float
            "depth": int(self.depth),
            "children_count": len(self.children),
            "children": []
        }

        if max_depth is None or level < max_depth:
            sorted_children = sorted(self.children, key=lambda x: x.visit_count, reverse=True)
            node_data["children"] = [child._get_tree_data(max_depth, level + 1) for child in sorted_children]

        return node_data


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad
    def search(self, state, player):
        root = Node(self.game, self.args, state, player, visit_count=1)
        
        policy, value = self.model(
            torch.tensor(self.game.get_encoded_state(state, player), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        
        de = self.args['dirichlet_epsilon']
        da = self.args['dirichlet_alpha']
        policy = (1 - de) * policy + de * np.random.dirichlet([da] * self.game.action_size)
        
        valid_modes = self.game.get_valid_moves(state, player)
        policy *= valid_modes
        policy /= sum(policy)
        root.expand(policy)
        
        for _ in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.player)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state, node.player), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state, node.player)
                policy *= valid_moves  # mask out invalid moves
                policy /= np.sum(policy)
              
                value = value.item()
                
                node = node.expand(policy)
                
            node.backpropagate(value)
        
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        
        # root.inspect()
        return action_probs


class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(4, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backbone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.col_count, game.action_size)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.col_count, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for res_block in self.backbone:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
    

class AlphaZero:
    def __init__(self, model, optimizer, game, args, initial_state=None):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.initial_state = initial_state
        
    def self_play(self, starting_player):
        memory = []
        if self.initial_state:
            state = self.initial_state.copy()
        else:
            state = self.game.get_initial_state()

        current_player = starting_player
        
        while True:
            # Check for terminal state FIRST using the game's built-in logic
            value, is_terminal = self.game.get_value_and_terminated(state, current_player)
            if is_terminal:
                break
                
            # Get action probabilities from MCTS
            action_probs = self.mcts.search(state, current_player)
            
            # Store the state from the perspective of the current player
            memory.append([state.copy(), action_probs.copy(), current_player])
            
            # Select action
            if np.sum(action_probs) == 0:
                # No valid moves - this should have been caught by termination check
                print("Warning: No valid moves but game not terminated")
                break
                
            # Use temperature if specified
            if self.args.get('use_temperature', False) and self.args['temperature'] != 1.0:
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                if np.sum(temperature_action_probs) > 0:
                    temperature_action_probs /= np.sum(temperature_action_probs)
                    action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                else:
                    action = np.random.choice(self.game.action_size, p=action_probs)
            else:
                action = np.random.choice(self.game.action_size, p=action_probs)
            
            # Apply the move
            state = self.game.get_next_state(state, action, current_player)
            
            # Switch players if not multi-jumping
            if not state.get('jump_again'):
                current_player = self.game.get_opponent(current_player)
        
        # At this point, the game has terminated naturally
        # Get the final game value from the perspective of the current player
        final_value, _ = self.game.get_value_and_terminated(state, current_player)
        
        # Create training examples
        return_memory = []
        for hist_state, hist_action_probs, hist_player in memory:
            # Calculate outcome from the perspective of hist_player
            if hist_player == current_player:
                hist_outcome = final_value
            else:
                hist_outcome = self.game.get_opponent_value(final_value)
            
            # Encode state from the perspective of the player who made the move
            encoded_state = self.game.get_encoded_state(hist_state, hist_player)
            
            return_memory.append((
                encoded_state,
                hist_action_probs,
                hist_outcome
            ))
        
        return return_memory
                
    def train(self, memory):
        if len(memory) == 0:
            return
            
        random.shuffle(memory)
        total_policy_loss = 0
        total_value_loss = 0
        batch_count = 0
        
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            batch_end = min(len(memory), batch_idx + self.args['batch_size'])
            sample = memory[batch_idx:batch_end]
            
            if len(sample) == 0:
                continue
                
            state, policy_targets, value_targets = zip(*sample)
            
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batch_count += 1
        
        return total_policy_loss / batch_count, total_value_loss / batch_count
    
    def learn(self):
        for i in tqdm(range(self.args['num_iterations']), desc="Training Iterations"):
            memory = []
            self.model.eval()
            
            # Alternate starting player each iteration for balance
            starting_player = 1 if i % 2 == 0 else -1
            
            game_outcomes = {'wins_p1': 0, 'wins_p_neg1': 0, 'draws': 0}
            successful_games = 0
            
            for game_idx in tqdm(range(self.args['num_self_play_iterations']), 
                               desc=f"Self-play games (starting with player {starting_player})"):
                try:
                    game_memory = self.self_play(starting_player)
                    if len(game_memory) > 0:
                        memory += game_memory
                        successful_games += 1
                        
                        # Track game outcomes for statistics
                        final_outcome = game_memory[-1][2]  # Last player's outcome
                        if final_outcome == 1:
                            if game_memory[-1][2] == 1:  # Player 1 won
                                game_outcomes['wins_p1'] += 1
                            else:  # Player -1 won
                                game_outcomes['wins_p_neg1'] += 1
                        elif final_outcome == 0:
                            game_outcomes['draws'] += 1
                            
                except Exception as e:
                    print(f"Error in self-play game {game_idx}: {e}")
                    continue
            
            # Print game statistics
            print(f"Iteration {i}: {successful_games} games completed")
            print(f"  Player 1 wins: {game_outcomes['wins_p1']}")
            print(f"  Player -1 wins: {game_outcomes['wins_p_neg1']}")
            print(f"  Draws: {game_outcomes['draws']}")
            print(f"  Total training samples: {len(memory)}")
            
            if len(memory) == 0:
                print("No training data collected, skipping training")
                continue
            
            # Training phase
            self.model.train()
            total_policy_loss = 0
            total_value_loss = 0
            
            for epoch in tqdm(range(self.args['num_epochs']), desc="Training epochs"):
                policy_loss, value_loss = self.train(memory)
                total_policy_loss += policy_loss
                total_value_loss += value_loss
            
            avg_policy_loss = total_policy_loss / self.args['num_epochs']
            avg_value_loss = total_value_loss / self.args['num_epochs']
            
            print(f"  Average Policy Loss: {avg_policy_loss:.4f}")
            print(f"  Average Value Loss: {avg_value_loss:.4f}")
            
            # Save model
            torch.save(self.model.state_dict(), f"saved_models/model_{i}.pt")
            torch.save(self.optimizer.state_dict(), f"saved_models/optimizer_{i}.pt")
            
            print(f"Model saved after iteration {i}\n")


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad
    def search(self, state, player):
        # Check if the state is already terminal
        value, is_terminal = self.game.get_value_and_terminated(state, player)
        if is_terminal:
            # Return zero policy for terminal states
            return np.zeros(self.game.action_size)
        
        root = Node(self.game, self.args, state, player, visit_count=1)
        
        policy, value = self.model(
            torch.tensor(self.game.get_encoded_state(state, player), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        
        # Add Dirichlet noise to root policy
        de = self.args['dirichlet_epsilon']
        da = self.args['dirichlet_alpha']
        policy = (1 - de) * policy + de * np.random.dirichlet([da] * self.game.action_size)
        
        # Mask invalid moves
        valid_moves = self.game.get_valid_moves(state, player)
        policy *= valid_moves
        if np.sum(policy) > 0:
            policy /= np.sum(policy)
        else:
            # No valid moves - this should have been caught above
            return np.zeros(self.game.action_size)
        
        root.expand(policy)
        
        for _ in range(self.args['num_searches']):
            node = root
            
            # Selection: traverse to leaf
            while node.is_fully_expanded():
                node = node.select()
                
            # Check if leaf is terminal using game logic
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.player)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                # Expansion: get policy and value from neural network
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state, node.player), 
                               device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state, node.player)
                policy *= valid_moves
                if np.sum(policy) > 0:
                    policy /= np.sum(policy)
                    value = value.item()
                    node = node.expand(policy)
                else:
                    # No valid moves from this node
                    value = 0
                    
            # Backpropagation
            node.backpropagate(value)
        
        # Generate action probabilities based on visit counts
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        
        if np.sum(action_probs) > 0:
            action_probs /= np.sum(action_probs)
        
        return action_probs


if __name__ == "__main__":
    checkers = Checkers()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ResNet(checkers, 8, 64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    args = {
        'C': 2,
        'num_searches': 50,
        'num_iterations': 20,
        'num_self_play_iterations': 25,
        'num_epochs': 10,
        'batch_size': 64,
        'temperature': 1.0,
        'use_temperature': False,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.6,
    }

    # Create saved_models directory if it doesn't exist
    import os
    os.makedirs("saved_models", exist_ok=True)

    initial_state = None
    
    az = AlphaZero(model, optimizer, checkers, args, initial_state=initial_state)
    az.mcts = MCTS(checkers, args, model)
    
    az.learn()