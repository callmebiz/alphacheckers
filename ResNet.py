# ResNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device, use_checkpoint=False):
        super().__init__()
        self.device = device
        self.use_checkpoint = use_checkpoint
        
        # Store architecture parameters for checkpointing
        self.num_resBlocks = num_resBlocks
        self.num_hidden = num_hidden
        
        # Get the number of input channels from the game
        input_channels = game.get_num_channels()
        print(f"ResNet input channels: {input_channels}")
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(input_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        # Create backbone with optional gradient checkpointing
        self.backbone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        # Apply gradient checkpointing if requested and we have multiple blocks
        if use_checkpoint and num_resBlocks > 1:
            print(f"Using gradient checkpointing for {num_resBlocks} ResBlocks")
            # Convert to sequential for checkpointing
            self.backbone_sequential = nn.Sequential(*self.backbone)
            self.use_sequential_checkpoint = True
        else:
            self.use_sequential_checkpoint = False
        
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
        
        # Apply backbone with or without checkpointing
        if self.use_sequential_checkpoint and self.training:
            # Use gradient checkpointing during training
            segments = min(2, len(self.backbone))  # Split into segments
            x = torch.utils.checkpoint.checkpoint_sequential(
                self.backbone_sequential, segments, x, use_reentrant=False
            )
        else:
            # Standard forward pass
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