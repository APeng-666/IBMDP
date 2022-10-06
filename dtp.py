'''
1, Training procedure in IBMDP
2, Extract decision tree policy from current agent 'DTP'
'''

from replay_buffer import ReplayBuffer  
from networks import QNetwork, DuelingQNetwork
from nodes import InternalNode, LeafNode

import torch
import torch.nn as nn 
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import random
import numpy as np

class DTP():
    """Interacts with and learns from the iteratively bounding environment."""

    def __init__(self,
                 ibenv,
                 device,
                 args):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            Network (str): dqn network type
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.seed = args.seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.ibenv = ibenv
        self.state_size = self.ibenv.observation_space.shape[0]
        self.action_size = self.ibenv.action_space.n
        self.layer_size = args.layer_size

        self.device = device
        self.LR = args.learning_rate
        self.TAU = args.tau
        self.GAMMA = args.gamma
        self.UPDATE_EVERY = args.update_every
        self.NUPDATES = args.n_updates
        self.BATCH_SIZE = args.batch_size
        self.BUFFER_SIZE = args.buffer_size
        self.Q_updates = 0
        self.clip_grad = args.clip_grad
        self.max_ep_len = args.max_ep_len


        # Q-Network
        self.qnetwork_dtp = DuelingQNetwork(self.ibenv.features_dim *2, self.ibenv.action_space.n, self.layer_size, self.seed).to(self.device)
        self.qnetwork_omi = DuelingQNetwork(self.state_size, self.action_size, self.layer_size, self.seed).to(self.device)
        self.qnetwork_target = DuelingQNetwork(self.state_size, self.action_size, self.layer_size, self.seed).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_omi.state_dict())

        self.optimizer_dtp = optim.Adam(self.qnetwork_dtp.parameters(), lr= self.LR)
        self.optimizer_omi = optim.Adam(self.qnetwork_omi.parameters(), lr= self.LR)
        print(self.qnetwork_dtp)
        print(self.qnetwork_omi)
        
        # Replay memory
        print("Using Regular Experience Replay")
        self.replay_buffer = ReplayBuffer(buffer_size=self.BUFFER_SIZE,
                                       batch_size= self.BATCH_SIZE,
                                       seed= self.seed, device = self.device)
        
        # define loss
        if args.loss == 'mse':
            self.loss = nn.MSELoss()
        elif args.loss == 'huber':
            self.loss = nn.SmoothL1Loss()
        else:
            print("Loss is not defined choose between mse and huber!")
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.t_step_omi = 0
        self.delay_omi_update = 2
        self.dtp_times = 0

        self.Q_loss_dtp = []
        self.Q_loss_omi = []
    
    #######################################
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > self.BATCH_SIZE:
                #Q_losses = []
                for _ in range(self.NUPDATES):
                    experiences = self.replay_buffer.sample()
                    #loss_dtp, loss_omi = self.learn(experiences)
                    loss_dtp = self.learn(experiences)
                    self.Q_updates += 1
                    self.Q_loss_dtp.append(loss_dtp)
                    #self.Q_loss_omi.append(loss_omi)
                    #Q_losses.append(loss)
                #.log({"Q_loss": np.mean(Q_losses), "Optimization step": self.Q_updates})
                    

    def get_action(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array or tensor): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if state.shape[-1] != self.ibenv.features_dim *2:
            # if not an unbase_state
            state = state[:, self.ibenv.features_dim:]
        self.qnetwork_dtp.eval()
        with torch.no_grad():
            action_values = self.qnetwork_dtp(state)
        self.qnetwork_dtp.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def get_actions(self, states):
        """Returns actions for given state as per current policy.
        state shape = [batch_num, state_size]

        Params
        ======
            state (array or tensor): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if not torch.is_tensor(states):
            states = torch.from_numpy(states).float().to(self.device)
        if states.shape[-1] != self.ibenv.features_dim *2:
            # if not an unbase_state
            states = states[:, self.ibenv.features_dim:]
        self.qnetwork_dtp.eval()
        with torch.no_grad():
            action_values = self.qnetwork_dtp(states)
        self.qnetwork_dtp.train()

        return torch.argmax(action_values, dim=1).unsqueeze(1)
        

    def learn(self, experiences):
        """Update parameters of q_dtp and q_omi using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            ## Compute and minimize the loss
            ### Extract next maximum estimated value from dtp network
            next_actions = self.get_actions(next_states)
            q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
            ### Calculate target value from bellman equation
            q_targets = rewards + self.GAMMA * q_targets_next * (1 - dones)

        ### Calculate expected value from local networks
        unbase_states = states[:, self.ibenv.features_dim :]
        q_expected_dtp = self.qnetwork_dtp(unbase_states).gather(1, actions)
        loss_dtp = self.loss(q_expected_dtp, q_targets)
        self.optimizer_dtp.zero_grad()
        loss_dtp.backward()
        self.optimizer_dtp.step()

        self.t_step_omi = (self.t_step_omi + 1) % self.delay_omi_update
        if self.t_step_omi == 0:
            q_expected_omi = self.qnetwork_omi(states).gather(1, actions)
            loss_omi = self.loss(q_expected_omi, q_targets)
            self.optimizer_omi.zero_grad()
            loss_omi.backward()
            self.optimizer_omi.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_omi, self.qnetwork_target)
        
        return loss_dtp.detach().cpu().numpy()#, loss_omi.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

    def save(self, filename= 'models/'):
        print("\nSaving network models...", end="")
        torch.save(self.qnetwork_dtp.state_dict(), filename + 'DTP')
        torch.save(self.optimizer_dtp.state_dict(), filename + 'DTP_optimizer')
        torch.save(self.qnetwork_omi.state_dict(), filename + 'OMI')
        torch.save(self.optimizer_omi.state_dict(), filename + 'OMI_optimizer')
        print("\nSuccess!", end="")


    def load(self, filename= 'models/'):
        print("\nLoading network models...", end="")
        self.qnetwork_dtp.load_state_dict(torch.load(filename + 'DTP'))
        self.optimizer_dtp.load_state_dict(torch.load(filename + 'DTP_optimizer'))
        self.qnetwork_omi.load_state_dict(torch.load(filename + 'OMI'))
        self.optimizer_omi.load_state_dict(torch.load(filename + 'OMI_optimizer'))
        self.qnetwork_target.load_state_dict(self.qnetwork_omi.state_dict())
        print("\nSuccess!", end="")

    def train(self, num_epochs, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        eps = eps_start
        record_steps = max(num_epochs // 500, 1)
        prev_score = -10000
        for t in range(num_epochs):
            total_score = 0
            traverse_times = 0
            state = self.ibenv.reset()
            for h in range(self.max_ep_len):
                action = self.get_action(state,eps)
                next_state, reward, done, info = self.ibenv.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                total_score += reward
                if info['traverse_tree']:
                    traverse_times += 1
                if done:
                    break
            if (t+1)% record_steps == 0:
                print('\nEpisode [{}/{}] \tTotal Score: {:.2f} \tTree decision times: {} \tLength of Episode: {}'.format(t+1, num_epochs, total_score,traverse_times, h+1), end="")
            if total_score > prev_score:
                self.save()
                prev_score = total_score
            eps = max(eps_end, eps_decay*eps) 
        return
    
    def subtree_from_policy(self, unbase_state):
        self.dtp_times +=1
        if self.dtp_times > 8:
            rand_act = np.random.randint(0,self.ibenv.base_action_n)
            return LeafNode(rand_act)
        action = self.get_action(unbase_state)
        act_i, act_j = self.ibenv.id_to_action[action]
        if act_j:
            #print("------------------")
            #print(act_i, act_j)
            unnorm_feature = act_j * (unbase_state[self.ibenv.features_dim + act_i] - unbase_state[act_i]) \
                                      + unbase_state[act_i]
            #print(unbase_state[self.ibenv.features_dim + act_i], unbase_state[act_i])
            #print(unnorm_feature)
            #print("------------------")
            unbase_state_L = np.copy(unbase_state)
            unbase_state_R = np.copy(unbase_state)
            unbase_state_L[self.ibenv.features_dim + act_i] = unnorm_feature
            unbase_state_R[act_i] = unnorm_feature 
            child_L = self.subtree_from_policy(unbase_state_L)
            child_R = self.subtree_from_policy(unbase_state_R)
            return InternalNode(act_i, unnorm_feature, child_L, child_R)
        else:
            return LeafNode(act_i)

    def extract_dtp(self):
        unbase_state = np.concatenate((self.ibenv.features_low, self.ibenv.features_high))
        self.dtp_times = 0
        decision_tree_policy = self.subtree_from_policy(unbase_state)
        return decision_tree_policy


    
    

