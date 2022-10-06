import torch
import numpy as np
import time
import argparse
import gym

from wrapper import IBEnvDisWrapper
from dtp import DTP
from plot_record import plot_dtp, plot_losses

def timer(start,end):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def evaluate_dtp(env, decision_tree_policy, eval_runs, seed):
    env.seed(seed+1)
    scores = []
    node = decision_tree_policy
    for _ in range(eval_runs):
        actions=[]
        state = env.reset()
        score = 0
        done = False
        while not done:
            while not node.leaf:
                if state[node.feature] <= node.value:
                    node = node.child_L
                else:
                    node = node.child_R
            action = node.action
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
        scores.append(score)
        print(actions)
    return np.mean(np.array(scores))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-info", type=str, default="Experiment-1",
                     help="Name of the Experiment (default: Experiment-1)")
    parser.add_argument('-env', type=str, default="CartPole-v1",
                     help='Name of the environment (default: CartPole-v1)')
    parser.add_argument('-epi', "--episodes", type=int, default=500,
                     help='Number of training episodes (default: 1000)')    
    parser.add_argument('-mel', "--max_ep_len", type=int, default=2000,
                     help='Number of training episodes (default: 500)')   
    parser.add_argument("--eval_every", type=int, default=100,
                     help="Evaluate the current policy every X episodes (default: 100)")
    parser.add_argument("--eval_runs", type=int, default=1,
                     help="Number of evaluation runs to evaluate - averating the evaluation Performance over all runs (default: 3)")
    parser.add_argument('-buf', '--buffer_size', type=int, default=100000,
                     help='Replay buffer size (default: 100000)')
    parser.add_argument('-bat', "--batch_size", type=int, default=256,
                     help='Batch size (default: 128)')
    parser.add_argument('-l', "--layer_size", type=int, default=64,
                     help='Neural Network layer size (default: 64)')
    parser.add_argument('-g', "--gamma", type=float, default=1,
                     help='Discount factor gamma (default: 0.99)')
    parser.add_argument('-t', "--tau", type=float, default=0.005,
                     help='Soft update factor tau (default: 0.005)')
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3,
                     help='Learning rate (default: 1e-3)')
    parser.add_argument('-u', "--update_every", type=int, default=2,
                     help='update the network every x step (default: 4)')
    parser.add_argument('-n_up', "--n_updates", type=int, default=1,
                     help='update the network for x steps (default: 1)')
    parser.add_argument('-s', "--seed", type=int, default=0,
                     help='random seed (default: 666)')
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Clip gradients (default: 1.0)")
    parser.add_argument("--loss", type=str, choices=["mse", "huber"], default="mse", help="Choose loss type MSE or Huber loss (default: mse)")
    
    args = parser.parse_args()

    env = gym.make(args.env)
    test_env = gym.make(args.env)
    print(env.observation_space)
    print(env.action_space)

    if args.env == 'CartPole-v1':
        bound = np.array([2, 2, 0.14, 1.4])
    else:
        bound = 100
    ibmdp = IBEnvDisWrapper(env, 3, bound, -0.01, args.seed)
    print(ibmdp.observation_space)
    print(ibmdp.action_space)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    agent = DTP(ibmdp, device, args)

    
    t0 = time.time()
    agent.train(args.episodes)
    t1 = time.time()

    timer(t0, t1)
    agent.save('models/'+str(args.episodes))
        
    #agent.load('models/'+str(args.episodes))

    dtp = agent.extract_dtp()
    
    eval_score = evaluate_dtp(test_env, dtp, args.eval_runs, args.seed)
    print('\nEvaluated score of the extracted decision tree policy: ' + str(eval_score))
    plot_dtp(dtp)
    plot_losses(agent.Q_loss_dtp)
