"""Policy value network"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import CONFIG
from torch.cuda.amp import autocast
from game import availabel,State

class ResBlock(nn.Module):

    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm1d(num_filters, )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm1d(num_filters, )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        # print("y",y)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)

#Input：N, 6*61 ---> N,C,H
class Net(nn.Module):

    def __init__(self, num_channels=256, num_res_blocks=7):
        super().__init__()
        # Initialization feature
        self.conv_block = nn.Conv1d(in_channels=6, out_channels=num_channels, kernel_size= 3, stride= 1, padding=1)
        self.conv_block_bn = nn.BatchNorm1d(256)
        self.conv_block_act = nn.ReLU()
        # Residual block extraction feature
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
        # Policy module
        self.policy_conv = nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size= 1, stride= 1)
        self.policy_bn = nn.BatchNorm1d(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * 61, 2000)
        # Value module
        self.value_conv = nn.Conv1d(in_channels=num_channels, out_channels=8, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm1d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 61, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    # Define forward propagation
    def forward(self, x):
        # Common module
        # print("x.shape", x.shape)
        x = self.conv_block(x)
        # print("x.shape",x.shape)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)
        # Policy module
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, 16 * 61])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy)
        # Value module
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, 8 * 61])
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return policy, value

if __name__ == '__main__' :
    net = Net()
    test_data = torch.ones([8,6,61])
    # print(test_data)
    policy,value = net(test_data)
    print(policy.shape)
    print(value.shape)



# Policy value network is used to train the model
class PolicyValueNet:

    def __init__(self, model_file=None, use_gpu=True, device = 'cuda:1'):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3    # l2 regularization
        self.device = device
        self.policy_value_net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.l2_const)
        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))  # Loading model parameters

    # Input the state of a batch
    # output the action probability and state value of a batch
    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy())
        return act_probs, value.detach().numpy()

    # Input sequence, returns a list of tuples (action, probability)
    # Each legally assembled action, and a score for the sequence state
    def policy_value_fn(self,sequence, structure, list1, list2):
        self.policy_value_net.eval()
        # Gets a list of legal actions
        move,sequences,structures = availabel(sequence, structure, list1, list2)

        # Convert sequences and structures into tensors
        state = State(sequence, structure)
        current_state = np.ascontiguousarray(state.astype('float32'))
        current_state = torch.as_tensor(current_state).to(self.device)
        # Use neural networks to make predictions
        with autocast(): #Single precision fp32
            log_act_probs, value = self.policy_value_net(current_state)
        log_act_probs, value = log_act_probs.cpu() , value.cpu()
        probs = np.exp(log_act_probs.detach().numpy().astype('float32').flatten())
        # Only take out legal actions
        act_probs = zip(sequences,structures, probs[move])
        # Returns the probability of the action, and the value of the state
        return act_probs, value.detach().numpy()

    # Save model
    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

    # Save model
    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()

        # Packaging variable
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32).to(self.device)
        winner_batch = torch.tensor(winner_batch, dtype=torch.float32).to(self.device)
        # Clear gradient
        self.optimizer.zero_grad()
        # Set learning rate
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        # Forward operation
        log_act_probs, value = self.policy_value_net(state_batch)
        value = torch.reshape(value, shape=[-1])
        # Loss of value
        value_loss = F.mse_loss(input=value, target=winner_batch)
        # Strategy loss
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))  # two vectors to go in the same direction as possible
        # For the total loss, note that l2 penalties are already included within the optimizer
        loss = value_loss + policy_loss
        # Backpropagation and optimization
        loss.backward()
        self.optimizer.step()
        # The entropy of the calculation strategy is used only to evaluate the model
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()

if __name__ == '__main__' :
    sequence = 'AAACCCAAACCCCCCAAAAAA&GGG&GGGAAACCCCCC&GGG&GGG&GGGAAA'
    structure = '...(((...((((((......&)))&)))...((((((&)))&)))&)))...'
    list1 = ["AAA", "CCC&GGG"]
    list2 = ["...", "(((&)))"]
