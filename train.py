import argparse
from itertools import count
import math
import random
import torch

import torch.nn as nn

from game import SnakeGame
from memory import ReplayMemory, Transition
from network import DQN

parser = argparse.ArgumentParser()
parser.add_argument('--no-resume', action='store_true', help='Start training from scratch, ignoring any saved model')
args = parser.parse_args()

device = torch.device('cpu')

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 1e-3

wnd = 2 * 2 + 1
n_scalar      = 12        # food(4) + danger(3) + direction(4) + body_len(1)
grid_size     = 7
n_observations = n_scalar + grid_size * grid_size  # 61
n_actions = 4

policy_net = DQN(n_observations, n_actions, n_scalar=n_scalar, grid_size=grid_size).to(device)
if args.no_resume:
    print("Starting from scratch (--no-resume).")
else:
    try:
        policy_net.load_state_dict(torch.load('best.pt', weights_only=True))
        print("Loaded pre-trained model from best.pt")
    except FileNotFoundError:
        print("No pre-trained model found, starting from scratch.")
    except RuntimeError as e:
        print(f"Error loading model: {e}. Starting from scratch.")

target_net = DQN(n_observations, n_actions, n_scalar=n_scalar, grid_size=grid_size).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100_000)

game = SnakeGame(wnd=wnd, field_size=(20, 20))
game.reset()

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        policy_net.eval()
        with torch.no_grad():
            action = policy_net(state).max(1).indices.view(1, 1)
        policy_net.train()
        return action
    else:
        return torch.tensor([[random.randint(0,3)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch)
    state_action_values = state_action_values.gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss


num_episodes = 25000

best_len = 0

for i_episode in range(num_episodes):
    game.reset(random_field_size=True)
    state = game.observation()

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated = game.update(action.item())

        reward = torch.tensor([reward], device=device)
        done = terminated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        loss = optimize_model()

        # Soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            if len(game.gamestate.body) > best_len:
                best_len = len(game.gamestate.body)
                torch.save(policy_net.state_dict(), 'best.pt')
                memory.save('memory.pickle')
                print(best_len)

            if i_episode % 10 == 0 and i_episode > 0:
                print(f"Episode {i_episode}, Step {t}, Loss: {loss.item() if loss is not None else 'N/A'}, Best Length: {best_len}")

                policy_net.eval()
                total_reward = 0
                eval_best_len = 0
                for _ in range(100):
                    eval_game = SnakeGame(wnd=wnd, field_size=(20, 20))
                    eval_game.reset(random_field_size=True)
                    while True:
                        eval_state = torch.tensor(eval_game.observation(), dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            eval_action = policy_net(eval_state).max(1).indices.view(1, 1)
                        _, _, eval_terminated = eval_game.update(eval_action.item())
                        if eval_terminated:
                            total_reward += eval_game.gamestate.reward
                            eval_best_len = max(eval_best_len, len(eval_game.gamestate.body))
                            break
                policy_net.train()
                print(f"Average reward: {total_reward / 100}, Eval best len: {eval_best_len}")

            break
