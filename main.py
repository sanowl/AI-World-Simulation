import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import List, Tuple
import heapq
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ultra-Advanced AI World Simulation V2")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Define a Tile class for the world grid
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
        
        # Apply pollution effect
        pollution_factor = min(self.pollution / 100, 1)
        color = tuple(int(c * (1 - pollution_factor) + 100 * pollution_factor) for c in color)
        
        pygame.draw.rect(screen, color, (x, y, size, size))
        if self.resource > 0:
            pygame.draw.circle(screen, YELLOW, (x + size // 2, y + size // 2), size // 4)

# World class
class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[Tile('grass', True) for _ in range(width)] for _ in range(height)]
        self.generate_world()
        self.weather = 'clear'
        self.weather_timer = 0

    def generate_world(self):
        # Generate water bodies
        for _ in range(5):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.generate_blob(x, y, 'water', 0.2)

        # Generate forests
        for _ in range(10):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.generate_blob(x, y, 'forest', 0.15)

        # Generate mountains
        for _ in range(3):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.generate_blob(x, y, 'mountain', 0.1)

        # Generate deserts
        for _ in range(2):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.generate_blob(x, y, 'desert', 0.1)

        # Generate cities
        for _ in range(5):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            if self.grid[y][x].type == 'grass':
                self.grid[y][x] = Tile('city', True)

        # Add resources
        for _ in range(50):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            if self.grid[y][x].type in ['grass', 'forest', 'desert']:
                self.grid[y][x].resource = random.randint(50, 100)

    def generate_blob(self, x: int, y: int, tile_type: str, size_factor: float):
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

    def draw(self, screen):
        tile_size = min(WIDTH // self.width, HEIGHT // self.height)
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x].draw(screen, x * tile_size, y * tile_size, tile_size)

        # Draw weather effect
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

# Agent class
class Agent:
    def __init__(self, x: int, y: int, agent_type: str):
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
            'crafting': random.random()
        }

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

    def rest(self):
        self.energy = min(100, self.energy + 10)
        self.health = min(100, self.health + 5)

    def craft(self):
        if self.inventory >= 10:
            self.inventory -= 10
            self.skills['crafting'] += 0.05
            self.energy -= 5

    def draw(self, screen, tile_size: int):
        color = RED if self.type == 'explorer' else BLUE if self.type == 'gatherer' else PURPLE
        pygame.draw.circle(screen, color, 
                           (self.x * tile_size + tile_size // 2, 
                            self.y * tile_size + tile_size // 2), 
                           tile_size // 3)

        # Draw health bar
        health_width = int(tile_size * self.health / 100)
        pygame.draw.rect(screen, GREEN, (self.x * tile_size, self.y * tile_size - 5, health_width, 3))

    def update(self):
        self.age += 1
        if self.age % 100 == 0:
            self.health -= 1

# Neural Network for the agent's brain
class AgentNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(AgentNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
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

# DQN Agent
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

    def act(self, state: np.ndarray) -> int:
        if random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_values = self.policy_net(state)
            return action_values.argmax().item()

    def learn(self):
        if len(self.memory) < 64:
            return

        experiences = self.memory.sample(64)
        batch = Experience(*zip(*experiences))

        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# A* Pathfinding
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(world: World, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    open_set = []
    heapq.heappush(open_set, (fscore[start], start))
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        close_set.add(current)
        
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if 0 <= neighbor[0] < world.width and 0 <= neighbor[1] < world.height:
                if not world.grid[neighbor[1]][neighbor[0]].passable:
                    continue
                tentative_g_score = gscore[current] + heuristic(current, neighbor)
                
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue


# Simulation class
class Simulation:
    def __init__(self, world_width: int, world_height: int, num_agents: int):
        self.world = World(world_width, world_height)
        self.agents = []
        self.dqn_agents = []
        self.step_count = 0
        
        for _ in range(num_agents):
            x, y = random.randint(0, world_width - 1), random.randint(0, world_height - 1)
            while not self.world.grid[y][x].passable:
                x, y = random.randint(0, world_width - 1), random.randint(0, world_height - 1)
            agent_type = random.choice(['explorer', 'gatherer', 'crafter'])
            self.agents.append(Agent(x, y, agent_type))
            self.dqn_agents.append(DQNAgent(12, 7))  # 12 state variables, 7 possible actions

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
            1 if tile.type == 'city' else 0,
            nearest_resource / (self.world.width + self.world.height) if nearest_resource else 1,
            nearest_city / (self.world.width + self.world.height) if nearest_city else 1,
            tile.pollution / 100
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

        for agent, dqn_agent in zip(self.agents, self.dqn_agents):
            state = self.get_state(agent)
            action = dqn_agent.act(state)
            
            # Execute action
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
            
            # Calculate reward
            reward = self.calculate_reward(agent)
            
            # Get new state
            next_state = self.get_state(agent)
            
            # Store experience in memory
            dqn_agent.memory.push(state, action, reward, next_state, False)
            
            # Learn from experience
            dqn_agent.learn()

            # Update agent
            agent.update()
        
        # Update target networks periodically
        if self.step_count % 100 == 0:
            for dqn_agent in self.dqn_agents:
                dqn_agent.update_target_network()
        
        self.step_count += 1

    def calculate_reward(self, agent: Agent) -> float:
        reward = 0
        tile = self.world.grid[agent.y][agent.x]

        # Basic survival reward
        reward += 0.1 if agent.health > 50 else -0.1
        reward += 0.1 if agent.energy > 50 else -0.1

        # Type-specific rewards
        if agent.type == 'explorer':
            if tile.type not in ['grass', 'city']:
                reward += 0.2  # Reward for exploring diverse terrain
        elif agent.type == 'gatherer':
            reward += 0.1 * agent.inventory  # Reward for holding resources
        elif agent.type == 'crafter':
            reward += agent.skills['crafting'] * 0.5  # Reward for high crafting skill

        # Resource management
        if agent.inventory > 0 and tile.type == 'city':
            reward += agent.inventory * 0.2  # Reward for bringing resources to city
            agent.inventory = 0  # Empty inventory in city

        # Environmental factors
        reward -= tile.pollution * 0.1  # Penalty for being in polluted areas

        # Skill improvement
        reward += sum(agent.skills.values()) * 0.1  # Small reward for overall skill improvement

        return reward

    def draw(self, screen):
        self.world.draw(screen)
        tile_size = min(WIDTH // self.world.width, HEIGHT // self.world.height)
        for agent in self.agents:
            agent.draw(screen, tile_size)

# Main game loop
def main():
    simulation = Simulation(world_width=60, world_height=40, num_agents=20)
    font = pygame.font.Font(None, 36)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        simulation.step()

        screen.fill(BLACK)
        simulation.draw(screen)

        # Display step count and weather
        step_text = font.render(f"Step: {simulation.step_count}", True, WHITE)
        screen.blit(step_text, (10, 10))
        weather_text = font.render(f"Weather: {simulation.world.weather}", True, WHITE)
        screen.blit(weather_text, (10, 50))

        pygame.display.flip()
        clock.tick(30)  # Limit to 30 FPS for better observation

    pygame.quit()

if __name__ == "__main__":
    main()
