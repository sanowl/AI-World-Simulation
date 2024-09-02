import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import List, Tuple, Dict
import heapq
import math
from scipy.spatial import cKDTree


pygame.init()


WIDTH, HEIGHT = 1600, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced AI World Simulation V3")


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)


clock = pygame.time.Clock()

class Tile:
    def __init__(self, type: str, passable: bool, resource: int = 0):
        self.type = type
        self.passable = passable
        self.resource = resource
        self.pollution = 0

    def draw(self, screen, x, y, size):
        color = {
            'grass': (34, 139, 34),
            'water': (0, 191, 255),
            'mountain': (139, 137, 137),
            'forest': (0, 100, 0),
            'city': (169, 169, 169),
            'desert': (244, 164, 96)
        }.get(self.type, WHITE)
        
        pollution_factor = min(self.pollution / 100, 1)
        color = tuple(int(c * (1 - pollution_factor) + 100 * pollution_factor) for c in color)
        
        pygame.draw.rect(screen, color, (x, y, size, size))
        if self.resource > 0:
            pygame.draw.circle(screen, YELLOW, (x + size // 2, y + size // 2), size // 4)

class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[Tile('grass', True) for _ in range(width)] for _ in range(height)]
        self.generate_world()
        self.weather = 'clear'
        self.weather_timer = 0
        self.agents = []
        self.kdtree = None

    def generate_world(self):
        for _ in range(5):
            self.generate_blob('water', 0.2)
        for _ in range(10):
            self.generate_blob('forest', 0.15)
        for _ in range(3):
            self.generate_blob('mountain', 0.1)
        for _ in range(2):
            self.generate_blob('desert', 0.1)
        for _ in range(5):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            if self.grid[y][x].type == 'grass':
                self.grid[y][x] = Tile('city', True)
        for _ in range(50):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            if self.grid[y][x].type in ['grass', 'forest', 'desert']:
                self.grid[y][x].resource = random.randint(50, 100)

    def generate_blob(self, tile_type: str, size_factor: float):
        x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
        queue = [(x, y)]
        visited = set()
        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) in visited or random.random() > size_factor:
                continue
            visited.add((cx, cy))
            if 0 <= cx < self.width and 0 <= cy < self.height:
                self.grid[cy][cx] = Tile(tile_type, tile_type not in ['water', 'mountain'])
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    queue.append((cx + dx, cy + dy))

    def update_weather(self):
        self.weather_timer -= 1
        if self.weather_timer <= 0:
            self.weather = random.choice(['clear', 'rainy', 'stormy'])
            self.weather_timer = random.randint(100, 300)

    def apply_weather_effects(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.weather == 'rainy':
                    if self.grid[y][x].type == 'desert':
                        self.grid[y][x].resource = min(100, self.grid[y][x].resource + 1)
                elif self.weather == 'stormy':
                    if self.grid[y][x].type in ['forest', 'grass']:
                        self.grid[y][x].resource = max(0, self.grid[y][x].resource - 1)

    def draw(self, screen, camera):
        tile_size = min(WIDTH // self.width, HEIGHT // self.height)
        for y in range(self.height):
            for x in range(self.width):
                screen_x = x * tile_size - camera[0]
                screen_y = y * tile_size - camera[1]
                if 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
                    self.grid[y][x].draw(screen, screen_x, screen_y, tile_size)

        if self.weather == 'rainy':
            for _ in range(100):
                rx = random.randint(0, WIDTH)
                ry = random.randint(0, HEIGHT)
                pygame.draw.line(screen, (200, 200, 255), (rx, ry), (rx, ry + 10), 1)
        elif self.weather == 'stormy':
            for _ in range(5):
                rx = random.randint(0, WIDTH)
                ry = random.randint(0, HEIGHT)
                pygame.draw.line(screen, (255, 255, 0), (rx, ry), (rx + random.randint(-20, 20), ry + random.randint(-20, 20)), 2)

    def update_kdtree(self):
        self.kdtree = cKDTree([(agent.x, agent.y) for agent in self.agents])

class Agent:
    def __init__(self, x: int, y: int, agent_type: str, communication_range: int = 5):
        self.x = x
        self.y = y
        self.type = agent_type
        self.inventory = 0
        self.health = 100
        self.energy = 100
        self.age = 0
        self.skills = {
            'gathering': random.random(),
            'exploration': random.random(),
            'crafting': random.random(),
            'trading': random.random()
        }
        self.communication_range = communication_range
        self.memory = deque(maxlen=100)  # Limit memory size
        self.mood = "neutral"
        self.long_term_goal = random.choice(["resource_master", "explorer", "trader"])
        self.goal_progress = 0

    def move(self, dx: int, dy: int, world: World):
        new_x, new_y = self.x + dx, self.y + dy
        if 0 <= new_x < world.width and 0 <= new_y < world.height and world.grid[new_y][new_x].passable:
            self.x, self.y = new_x, new_y
            self.energy -= 1 + world.grid[new_y][new_x].pollution * 0.1

    def gather_resource(self, world: World):
        tile = world.grid[self.y][self.x]
        if tile.resource > 0:
            gathered = min(tile.resource, 10 * (1 + self.skills['gathering']))
            tile.resource -= gathered
            self.inventory += gathered
            self.energy -= 2
            tile.pollution += 1
            if self.long_term_goal == "resource_master":
                self.goal_progress += gathered

    def rest(self):
        self.energy = min(100, self.energy + 10)
        self.health = min(100, self.health + 5)

    def craft(self):
        if self.inventory >= 10:
            self.inventory -= 10
            self.skills['crafting'] += 0.05
            self.energy -= 5

    def communicate(self, world: World):
        nearby_agents = world.kdtree.query_ball_point((self.x, self.y), self.communication_range)
        for idx in nearby_agents:
            agent = world.agents[idx]
            if agent != self:
                message = {
                    'x': self.x, 'y': self.y, 'type': self.type,
                    'inventory': self.inventory, 'health': self.health,
                    'mood': self.mood, 'goal': self.long_term_goal
                }
                agent.receive_message(message)

    def receive_message(self, message: dict):
        self.memory.append(message)
        if message['type'] == 'gatherer' and message['inventory'] > 50:
            self.mood = 'interested'
        if message['goal'] == self.long_term_goal and message['type'] != self.type:
            self.mood = 'competitive'

    def trade(self, other_agent):
        if self.inventory > 0 and other_agent.inventory > 0:
            trade_amount = min(self.inventory, other_agent.inventory, 10)
            self.inventory -= trade_amount
            other_agent.inventory -= trade_amount
            self.inventory += trade_amount
            other_agent.inventory += trade_amount
            self.skills['trading'] += 0.05
            other_agent.skills['trading'] += 0.05
            if self.long_term_goal == "trader":
                self.goal_progress += trade_amount

    def reproduce(self, world: World):
        if self.health > 80 and self.energy > 80:
            offspring = Agent(self.x, self.y, self.type)
            for skill in self.skills:
                offspring.skills[skill] = (self.skills[skill] + random.random()) / 2
            offspring.long_term_goal = random.choice(["resource_master", "explorer", "trader"])
            world.agents.append(offspring)
            self.energy -= 50

    def draw(self, screen, tile_size: int, camera):
        color = RED if self.type == 'explorer' else BLUE if self.type == 'gatherer' else PURPLE
        screen_x = self.x * tile_size - camera[0]
        screen_y = self.y * tile_size - camera[1]
        if 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
            pygame.draw.circle(screen, color, (screen_x + tile_size // 2, screen_y + tile_size // 2), tile_size // 3)
            health_width = int(tile_size * self.health / 100)
            pygame.draw.rect(screen, GREEN, (screen_x, screen_y - 5, health_width, 3))

    def update(self, world: World):
        self.age += 1
        if self.age % 100 == 0:
            self.health -= 1
        if self.long_term_goal == "explorer":
            if world.grid[self.y][self.x].type not in ['grass', 'city']:
                self.goal_progress += 1
        self.energy = max(0, self.energy - 0.1)
        if self.energy <= 0 or self.health <= 0:
            return False
        return True

class AgentNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(AgentNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x, hidden

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = AgentNN(state_size, 128, action_size).to(self.device)
        self.target_net = AgentNN(state_size, 128, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.hidden = None

    def act(self, state: np.ndarray) -> int:
        if random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            action_values, self.hidden = self.policy_net(state, self.hidden)
            return action_values.max(1)[1].item()

    def learn(self):
        if len(self.memory) < 64:
            return

        experiences = self.memory.sample(64)
        batch = Experience(*zip(*experiences))

        state_batch = torch.FloatTensor(batch.state).unsqueeze(1).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        q_values, _ = self.policy_net(state_batch, None)
        q_values = q_values.gather(1, action_batch)

        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_batch, None)
            next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class Simulation:
    def __init__(self, world_width: int, world_height: int, num_agents: int):
        self.world = World(world_width, world_height)
        self.agents = []
        self.dqn_agents = []
        self.step_count = 0
        self.camera = [0, 0]
        self.zoom = 1.0
        
        for _ in range(num_agents):
            x, y = random.randint(0, world_width - 1), random.randint(0, world_height - 1)
            while not self.world.grid[y][x].passable:
                x, y = random.randint(0, world_width - 1), random.randint(0, world_height - 1)
            agent_type = random.choice(['explorer', 'gatherer', 'crafter'])
            agent = Agent(x, y, agent_type)
            self.agents.append(agent)
            self.world.agents.append(agent)
            self.dqn_agents.append(DQNAgent(16, 8))  # 16 state variables, 8 possible actions

    def get_state(self, agent: Agent) -> np.ndarray:
        tile = self.world.grid[agent.y][agent.x]
        nearest_resource = self.find_nearest_resource(agent)
        nearest_city = self.find_nearest_city(agent)
        return np.array([
            agent.x / self.world.width,
            agent.y / self.world.height,
            agent.inventory / 100,
            agent.health / 100,
            agent.energy / 100,
            agent.skills['gathering'],
            agent.skills['exploration'],
            agent.skills['crafting'],
            agent.skills['trading'],
            1 if tile.type == 'city' else 0,
            nearest_resource / (self.world.width + self.world.height) if nearest_resource else 1,
            nearest_city / (self.world.width + self.world.height) if nearest_city else 1,
            tile.pollution / 100,
            agent.goal_progress / 1000,
            len(agent.memory) / 100,
            {'neutral': 0, 'happy': 1, 'interested': 2, 'competitive': 3}[agent.mood]
        ])

    def find_nearest_resource(self, agent: Agent) -> float:
        min_distance = float('inf')
        for y in range(self.world.height):
            for x in range(self.world.width):
                if self.world.grid[y][x].resource > 0:
                    distance = math.sqrt((agent.x - x)**2 + (agent.y - y)**2)
                    min_distance = min(min_distance, distance)
        return min_distance if min_distance != float('inf') else None

    def find_nearest_city(self, agent: Agent) -> float:
        min_distance = float('inf')
        for y in range(self.world.height):
            for x in range(self.world.width):
                if self.world.grid[y][x].type == 'city':
                    distance = math.sqrt((agent.x - x)**2 + (agent.y - y)**2)
                    min_distance = min(min_distance, distance)
        return min_distance if min_distance != float('inf') else None

    def step(self):
        self.world.update_weather()
        self.world.apply_weather_effects()
        self.world.update_kdtree()

        for agent, dqn_agent in zip(self.agents[:], self.dqn_agents):
            state = self.get_state(agent)
            action = dqn_agent.act(state)
            
            if action == 0:  # Move North
                agent.move(0, -1, self.world)
            elif action == 1:  # Move South
                agent.move(0, 1, self.world)
            elif action == 2:  # Move East
                agent.move(1, 0, self.world)
            elif action == 3:  # Move West
                agent.move(-1, 0, self.world)
            elif action == 4:  # Gather resource
                agent.gather_resource(self.world)
            elif action == 5:  # Rest
                agent.rest()
            elif action == 6:  # Craft
                agent.craft()
            elif action == 7:  # Trade
                nearby_agents = self.world.kdtree.query_ball_point((agent.x, agent.y), 1)
                for idx in nearby_agents:
                    other_agent = self.world.agents[idx]
                    if other_agent != agent:
                        agent.trade(other_agent)
                        break
            
            agent.communicate(self.world)
            agent.reproduce(self.world)
            
            reward = self.calculate_reward(agent)
            next_state = self.get_state(agent)
            
            dqn_agent.memory.push(state, action, reward, next_state, False)
            dqn_agent.learn()

            if not agent.update(self.world):
                self.agents.remove(agent)
                self.world.agents.remove(agent)
        
        if self.step_count % 100 == 0:
            for dqn_agent in self.dqn_agents:
                dqn_agent.update_target_network()
        
        self.step_count += 1

    def calculate_reward(self, agent: Agent) -> float:
        reward = 0
        tile = self.world.grid[agent.y][agent.x]

        reward += 0.1 if agent.health > 50 else -0.1
        reward += 0.1 if agent.energy > 50 else -0.1

        if agent.type == 'explorer':
            if tile.type not in ['grass', 'city']:
                reward += 0.2
        elif agent.type == 'gatherer':
            reward += 0.1 * agent.inventory
        elif agent.type == 'crafter':
            reward += agent.skills['crafting'] * 0.5

        if agent.inventory > 0 and tile.type == 'city':
            reward += agent.inventory * 0.2
            agent.inventory = 0

        reward -= tile.pollution * 0.1
        reward += sum(agent.skills.values()) * 0.1
        reward += agent.goal_progress * 0.01

        return reward

    def draw(self, screen):
        self.world.draw(screen, self.camera)
        tile_size = int(min(WIDTH // self.world.width, HEIGHT // self.world.height) * self.zoom)
        for agent in self.agents:
            agent.draw(screen, tile_size, self.camera)

        font = pygame.font.Font(None, 36)
        step_text = font.render(f"Step: {self.step_count}", True, WHITE)
        screen.blit(step_text, (10, 10))
        weather_text = font.render(f"Weather: {self.world.weather}", True, WHITE)
        screen.blit(weather_text, (10, 50))
        agents_text = font.render(f"Agents: {len(self.agents)}", True, WHITE)
        screen.blit(agents_text, (10, 90))

def main():
    simulation = Simulation(world_width=100, world_height=75, num_agents=50)
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    simulation.zoom *= 1.1
                elif event.key == pygame.K_MINUS:
                    simulation.zoom /= 1.1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    simulation.zoom *= 1.1
                elif event.button == 5:  # Scroll down
                    simulation.zoom /= 1.1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            simulation.camera[0] -= 5
        if keys[pygame.K_RIGHT]:
            simulation.camera[0] += 5
        if keys[pygame.K_UP]:
            simulation.camera[1] -= 5
        if keys[pygame.K_DOWN]:
            simulation.camera[1] += 5

        if not paused:
            simulation.step()

        screen.fill(BLACK)
        simulation.draw(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
