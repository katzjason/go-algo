import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleResidualBlock(nn.Module):
    def __init__(self, channels):
        super(SimpleResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = out + residual
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
        self.res_block1 = SimpleResidualBlock(32)
        self.res_block2 = SimpleResidualBlock(32)
        
        self.policy_conv = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1)
        self.policy_fc1 = nn.Linear(in_features=(2 * 9 * 9), out_features=128)
        self.policy_fc2 = nn.Linear(in_features=128, out_features=81)
        
        self.value_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.value_fc1 = nn.Linear(in_features=(1 * 9 * 9), out_features=64)
        self.value_fc2 = nn.Linear(in_features=64, out_features=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.initial_conv(x)
        x = F.relu(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        policy = self.policy_conv(x)
        policy = F.relu(policy)
        policy = policy.view(batch_size, -1) # reshapes to (batch_size, 162)
        policy = self.policy_fc1(policy)
        policy = F.relu(policy)
        policy_logits = self.policy_fc2(policy)
        
        value = self.value_conv(x)
        value = F.relu(value)
        value = value.view(batch_size, -1) # reshapes to (batch_size, 81)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value) # ensures values are in range [-1, 1] (W/L)

        return policy_logits, value
        