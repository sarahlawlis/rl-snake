import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# game constants
BLOCK_SIZE = 20
GRID_SIZE = 20
WINDOW_SIZE = BLOCK_SIZE * GRID_SIZE
FPS = 10

# actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [[GRID_SIZE // 2, GRID_SIZE // 2]]
        self.food = [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
        self.direction = (1, 0)
        self.done = False
        self.score = 0
        return self.get_state()


    def step(self, action):
        # 0 = left, 1 = right, 2 = forward
        x, y = self.direction

        # left
        if action == 0:
            self.direction = (-y, x)
        #right
        elif action == 1:
            self.direction = (y, -x)
        # If action == 2 (Move Forward), keep the same direction

        # Move forward
        head = self.snake[0]
        new_head = [head[0] + self.direction[0], head[1] + self.direction[1]]

        # check collision with walls
        if not (0 <= new_head[0] < GRID_SIZE and 0 <= new_head[1] < GRID_SIZE):
            self.done = True
            reward = -10  # Penalty for collision
            return self.get_state(), reward, self.done

        # check collision with self
        if new_head in self.snake:
            self.done = True
            reward = -10  # Penalty for collision
            return self.get_state(), reward, self.done

        # Insert new head position
        self.snake.insert(0, new_head)

        # check if food is eaten
        if new_head == self.food:
            self.score += 1
            self.food = [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
            reward = 10  # Reward for eating food
        else:
            # remove tail segment
            self.snake.pop()
            reward = 0  # Neutral reward for normal movement

        return self.get_state(), reward, self.done


    def get_state(self):
        head = self.snake[0]
        x, y = self.direction

        point_straight = [head[0] + x, head[1] + y]
        point_left = [head[0] - y, head[1] + x]
        point_right = [head[0] + y, head[1] - x]

        # danger detection
        danger_straight = (
            point_straight in self.snake or
            not (0 <= point_straight[0] < GRID_SIZE and 0 <= point_straight[1] < GRID_SIZE)
        )
        danger_left = (
            point_left in self.snake or
            not (0 <= point_left[0] < GRID_SIZE and 0 <= point_left[1] < GRID_SIZE)
        )
        danger_right = (
            point_right in self.snake or
            not (0 <= point_right[0] < GRID_SIZE and 0 <= point_right[1] < GRID_SIZE)
        )

        # movement direction
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        # food location relative to head
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]

        state = [
            # danger ahead
            danger_straight,
            danger_left,
            danger_right,
            # current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            food_left,
            food_right,
            food_up,
            food_down
        ]

        return np.array(state, dtype=int)

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.window.fill((0, 0, 0))
        for x, y in self.snake:
            pygame.draw.rect(self.window, (0, 255, 0), (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.window, (255, 0, 0), (self.food[0] * BLOCK_SIZE, self.food[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
        self.clock.tick(FPS)


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, input_dim=11, output_dim=3, hidden_dim=32, lr=0.01, gamma=0.5, epsilon_decay=1.0):
        self.model = DQN(input_dim, hidden_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=500)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 1.0
        self.epsilon_decay = epsilon_decay

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32, device=device)
            done = torch.tensor(done, dtype=torch.float32, device=device)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state).detach()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        pass  # No need to update the target model if it's not being used

# class Agent:
#     def __init__(self, input_dim=11, output_dim=3, hidden_dim=32, lr=0.01, gamma=0.5, epsilon_decay=1.0):
#         self.model = DQN(input_dim, hidden_dim, output_dim)
#         self.target_model = None
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#         self.memory = deque(maxlen=500)
#         self.gamma = gamma 
#         self.epsilon = 1.0  
#         self.epsilon_min = 1.0 
#         self.epsilon_decay = epsilon_decay


#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randint(0, 2) 
#         state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         with torch.no_grad():
#             return torch.argmax(self.model(state)).item()

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def replay(self, batch_size):
#         if len(self.memory) < batch_size:
#             return

#         batch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in batch:
#             state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#             next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
#             target = reward
#             if not done:
#                 target += self.gamma * torch.max(self.model(next_state)).item()
#             target_f = self.model(state).detach()

#             if action < len(target_f[0]):
#                 target_f[0][action] = target
#             else:
#                 print(f"Invalid Action: {action}, Allowed: 0-{len(target_f[0])-1}")

#             self.optimizer.zero_grad()
#             loss = nn.MSELoss()(self.model(state), target_f)
#             loss.backward()
#             self.optimizer.step()

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#     def update_target_model(self):
#         pass  # No need to update the target model if it's not being used


import matplotlib.pyplot as plt

def train():
    env = SnakeGame()
    agent = Agent(input_dim=11, output_dim=3)
    episodes = 200
    batch_size = 64

    rewards = []
    cumulative_rewards = []

    plt.ion() 
    fig, ax = plt.subplots(figsize=(12, 6))
    reward_line, = ax.plot([], [], label="Reward per Game", color="blue", alpha=0.7)
    avg_reward_line, = ax.plot([], [], label="Average Cumulative Reward", color="orange", alpha=0.7)
    ax.set_xlim(0, episodes)
    ax.set_ylim(0, 50)
    ax.set_title("Training Performance: Reward and Average Cumulative Reward")
    ax.set_xlabel("Number of Games")
    ax.set_ylabel("Reward / Average Cumulative Reward")
    ax.legend()
    plt.grid(True)

    last_annotation = None

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {e+1}/{episodes}, Epsilon: {agent.epsilon:.4f}, Reward: {total_reward}")
                rewards.append(total_reward)
                cumulative_rewards.append(total_reward)
                break

            agent.replay(batch_size)
        agent.update_target_model()

        avg_cumulative_rewards = [sum(cumulative_rewards[:i]) / i for i in range(1, len(cumulative_rewards) + 1)]

        reward_line.set_data(range(1, len(rewards) + 1), rewards)
        avg_reward_line.set_data(range(1, len(avg_cumulative_rewards) + 1), avg_cumulative_rewards)

        ax.set_xlim(0, episodes)
        ax.set_ylim(0, max(max(rewards, default=0), max(avg_cumulative_rewards, default=0)) + 10)

        if last_annotation:
            last_annotation.remove()

        if len(avg_cumulative_rewards) > 0:
            last_annotation = ax.annotate(f"{avg_cumulative_rewards[-1]:.2f}",
                                           xy=(len(avg_cumulative_rewards), avg_cumulative_rewards[-1]),
                                           xytext=(len(avg_cumulative_rewards) - 10, avg_cumulative_rewards[-1] + 5),
                                           fontsize=10, color="orange")

        plt.pause(0.01)


    plt.ioff()
    plt.show()

    with open("rewards.txt", "w") as f:
        for reward in rewards:
            f.write(f"{reward}\n")


if __name__ == "__main__":
    train()
