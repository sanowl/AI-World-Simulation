from collections import deque
import pygame
import secrets

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
            'gathering': secrets.SystemRandom().random(),
            'exploration': secrets.SystemRandom().random(),
            'crafting': secrets.SystemRandom().random(),
            'trading': secrets.SystemRandom().random()
        }
        self.communication_range = communication_range
        self.memory = deque(maxlen=100)  # Limit memory size
        self.mood = "neutral"
        self.long_term_goal = secrets.choice(["resource_master", "explorer", "trader"])
        self.goal_progress = 0

    def move(self, dx: int, dy: int, world):
        new_x, new_y = self.x + dx, self.y + dy
        if 0 <= new_x < world.width and 0 <= new_y < world.height and world.grid[new_y][new_x].passable:
            self.x, self.y = new_x, new_y
            self.energy -= 1 + world.grid[new_y][new_x].pollution * 0.1

    def gather_resource(self, world):
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

    def communicate(self, world):
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

    def reproduce(self, world):
        if self.health > 80 and self.energy > 80:
            offspring = Agent(self.x, self.y, self.type)
            for skill in self.skills:
                offspring.skills[skill] = (self.skills[skill] + secrets.SystemRandom().random()) / 2
            offspring.long_term_goal = secrets.choice(["resource_master", "explorer", "trader"])
            world.agents.append(offspring)
            self.energy -= 50

    def draw(self, screen, tile_size: int, camera):
        color = (255, 0, 0) if self.type == 'explorer' else (0, 0, 255) if self.type == 'gatherer' else (128, 0, 128)
        screen_x = self.x * tile_size - camera[0]
        screen_y = self.y * tile_size - camera[1]
        if 0 <= screen_x < screen.get_width() and 0 <= screen_y < screen.get_height():
            pygame.draw.circle(screen, color, (screen_x + tile_size // 2, screen_y + tile_size // 2), tile_size // 3)
            health_width = int(tile_size * self.health / 100)
            pygame.draw.rect(screen, (0, 255, 0), (screen_x, screen_y - 5, health_width, 3))

    def update(self, world):
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
