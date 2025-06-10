import code

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import json

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
        
    def self_play(self, player):
        memory = []
        if self.initial_state:
            state = self.initial_state
        else:
            state = self.game.get_initial_state()

        while True:
            action_probs = self.mcts.search(state, player)
            
            memory.append([state, action_probs, player])
            
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            # action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=action_probs)
            
            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state, player)
            if is_terminal:
                return_memory = []
                for hist_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    return_memory.append((
                        self.game.get_encoded_state(hist_state, player),
                        hist_action_probs,
                        hist_outcome
                    ))
                return return_memory
                    
            if not state['jump_again']:
                player = self.game.get_opponent(player)  # switch player unless multi-jumping
                
    def train(self, memory):
        random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            batch_end = min(len(memory) - 1, batch_idx + self.args['batch_size'])
            sample = memory[batch_idx:batch_end]
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
    
    def learn(self):
        player = 1
        for i in tqdm(range(self.args['num_iterations']), desc="num_iterations"):
            memory = []
            self.model.eval()
            for _ in tqdm(range(self.args['num_self_play_iterations']), desc="num_self_play_iterations"):
                memory += self.self_play(player)
            print(f"Training on {len(memory)} samples.")
            self.model.train()
            # code.interact(local=locals())
            for _ in tqdm(range(self.args['num_epochs']), desc="num_epochs"):
                self.train(memory)
                
            torch.save(self.model.state_dict(), f"saved_models/model_{i}.pt")
            torch.save(self.optimizer.state_dict(), f"saved_models/optimizer_{i}.pt")
        player = self.game.get_opponent(player)
            
            
if __name__ == "__main__":
    checkers = Checkers()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(checkers, 8, 64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    args = {
        'C': 2,
        'num_searches': 10,
        'num_iterations': 10,
        'num_self_play_iterations': 100,
        'num_epochs': 100,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': .6,
    }

    initial_state = None
    # initial_state, _ = checkers.load_state("state_allNans.txt")
    az = AlphaZero(model, optimizer, checkers, args, initial_state=initial_state)
    az.learn()
