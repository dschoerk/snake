import argparse
import math
import torch
import torch.nn as nn

from vec_game import VecSnakeGame
from memory import TensorReplayMemory
from network import DQN

parser = argparse.ArgumentParser()
parser.add_argument('--no-resume', action='store_true', help='Start training from scratch, ignoring any saved model')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 1024
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 1e-3
NUM_ENVS = 256
OPT_STEPS_PER_COLLECT = 4
EVAL_INTERVAL = 50
EVAL_GAMES = 100

wnd = 2 * 2 + 1
n_scalar = 12
grid_size = 7
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
memory = TensorReplayMemory(200_000, n_observations, device)
memory.load('memory.pickle')  # Load replay buffer if it exists, otherwise start fresh

vec_env = VecSnakeGame(NUM_ENVS, device=device)
steps_done = 0


def select_actions_batch(states_tensor):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    batch_size = states_tensor.shape[0]
    with torch.no_grad():
        greedy_actions = policy_net(states_tensor).max(1).indices

    random_mask = torch.rand(batch_size, device=device) < eps_threshold
    random_actions = torch.randint(0, n_actions, (batch_size,), device=device)
    return torch.where(random_mask, random_actions, greedy_actions)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, non_final_mask = memory.sample(BATCH_SIZE)

    state_action_values = policy_net(states).gather(1, actions)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_mask.any():
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(next_states[non_final_mask]).max(1).values
    expected = (next_state_values * GAMMA) + rewards

    loss = nn.functional.smooth_l1_loss(state_action_values, expected.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss


def run_eval():
    """Run greedy eval on GPU using VecSnakeGame."""
    eval_env = VecSnakeGame(EVAL_GAMES, device=device)
    policy_net.eval()
    states = eval_env.observations()
    alive = torch.ones(EVAL_GAMES, dtype=torch.bool, device=device)
    total_rewards = torch.zeros(EVAL_GAMES, device=device)
    final_lens = torch.ones(EVAL_GAMES, dtype=torch.int32, device=device)
    max_steps = 5000
    for _ in range(max_steps):
        if not alive.any():
            break
        with torch.no_grad():
            actions = policy_net(states).max(1).indices
        next_states, rewards, dones, body_lens = eval_env.step(actions)
        total_rewards += rewards * alive.float()
        just_died = dones & alive
        if just_died.any():
            final_lens[just_died] = body_lens[just_died]
        alive &= ~dones
        states = next_states
    policy_net.train()
    avg_reward = total_rewards.mean().item()
    eval_best_len = final_lens.max().item()
    return avg_reward, eval_best_len


num_episodes = 25000
best_len = 0
best_reward = float('-inf')
episode_count = 0
last_eval_episode = 0

# Initial observations — already on device
states_tensor = vec_env.observations()

while episode_count < num_episodes:
    actions = select_actions_batch(states_tensor)

    # Step all envs on GPU — no CPU involved
    next_states_tensor, rewards_tensor, dones_tensor, _ = vec_env.step(actions)

    # Push entire batch to replay buffer — all GPU tensors
    memory.push_batch(states_tensor, actions, rewards_tensor, next_states_tensor, dones_tensor)

    episode_count += dones_tensor.sum().item()

    # Eval check
    if episode_count - last_eval_episode >= EVAL_INTERVAL:
        last_eval_episode = episode_count

        avg_reward, eval_best_len = run_eval()
        last_loss = optimize_model()

        if eval_best_len > best_len or (eval_best_len == best_len and avg_reward > best_reward):
            best_len = eval_best_len
            best_reward = avg_reward
            torch.save(policy_net.state_dict(), 'best.pt')
            memory.save('memory.pickle')
            print(f"* New best saved: len={best_len}, reward={best_reward:.1f}")

        print(f"Episode {episode_count}, Loss: {last_loss.item() if last_loss is not None else 'N/A'}, Eval best len: {eval_best_len}, Avg reward: {avg_reward:.1f}")

    # Update states
    states_tensor = next_states_tensor

    # GPU optimization steps
    for _ in range(OPT_STEPS_PER_COLLECT):
        optimize_model()

    # Soft update target network
    with torch.no_grad():
        for p_target, p_policy in zip(target_net.parameters(), policy_net.parameters()):
            p_target.data.mul_(1 - TAU).add_(p_policy.data, alpha=TAU)
