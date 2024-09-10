import math
from typing import Callable, Dict, Optional, List
import pygame
import numpy as np
from world import World
from agent import Agent
from dqn_agent import DQNAgent
import secrets

class Simulation:
    def __init__(self, world_width: int, world_height: int, num_agents: int):
        self.world = World(world_width, world_height)
        self.agents: List[Agent] = []
        self.dqn_agents: List[DQNAgent] = []
        self.step_count = 0
        self.camera = [0, 0]
        self.zoom = 1.0
        
        self.agent_actions: Dict[int, Callable] = {
            0: lambda agent, world: agent.move(0, -1, world),  # Move North
            1: lambda agent, world: agent.move(0, 1, world),   # Move South
            2: lambda agent, world: agent.move(1, 0, world),   # Move East
            3: lambda agent, world: agent.move(-1, 0, world),  # Move West
            4: lambda agent, world: agent.gather_resource(world),
            5: lambda agent, _: agent.rest(),
            6: lambda agent, _: agent.craft(),
            7: self.trade_action
        }
        
        self.initialize_agents(num_agents, world_width, world_height)

    def initialize_agents(self, num_agents: int, world_width: int, world_height: int) -> None:
        for _ in range(num_agents):
            x, y = self.find_passable_location(world_width, world_height)
            agent_type = secrets.choice(['explorer', 'gatherer', 'crafter'])
            agent = Agent(x, y, agent_type)
            self.agents.append(agent)
            self.world.agents.append(agent)
            self.dqn_agents.append(DQNAgent(16, 8))  # 16 state variables, 8 possible actions

    def find_passable_location(self, width: int, height: int) -> tuple:
        while True:
            x, y = secrets.SystemRandom().randint(0, width - 1), secrets.SystemRandom().randint(0, height - 1)
            if self.world.grid[y][x].passable:
                return x, y

    def get_state(self, agent: Agent) -> np.ndarray:
        tile = self.world.grid[agent.y][agent.x]
        nearest_resource = self.find_nearest(agent, lambda x, y: self.world.grid[y][x].resource > 0)
        nearest_city = self.find_nearest(agent, lambda x, y: self.world.grid[y][x].type == 'city')
        
        mood_mapping = {'neutral': 0, 'happy': 1, 'interested': 2, 'competitive': 3}
        
        return np.array([
            agent.x / self.world.width,
            agent.y / self.world.height,
            agent.inventory / 100,
            agent.health / 100,
            agent.energy / 100,
            *(agent.skills.values()),
            1 if tile.type == 'city' else 0,
            nearest_resource / (self.world.width + self.world.height) if nearest_resource is not None else 1,
            nearest_city / (self.world.width + self.world.height) if nearest_city is not None else 1,
            tile.pollution / 100,
            agent.goal_progress / 1000,
            len(agent.memory) / 100,
            mood_mapping[agent.mood]
        ])

    def find_nearest(self, agent: Agent, condition: Callable[[int, int], bool]) -> Optional[float]:
        distances = (
            (x, y, math.hypot(agent.x - x, agent.y - y))
            for y in range(self.world.height)
            for x in range(self.world.width)
            if condition(x, y)
        )
        try:
            return min(distances, key=lambda t: t[2])[2]
        except ValueError:  # if distances is empty
            return None

    def step(self) -> None:
        self.world.update_weather()
        self.world.apply_weather_effects()
        self.world.update_kdtree()

        for agent, dqn_agent in list(zip(self.agents, self.dqn_agents)):
            state = self.get_state(agent)
            action = dqn_agent.act(state)
            
            self.agent_actions[action](agent, self.world)
            
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

    def trade_action(self, agent: Agent, world: World) -> None:
        if world.kdtree is not None:
            nearby_agents = world.kdtree.query_ball_point((agent.x, agent.y), 1)
            for idx in nearby_agents:
                other_agent = world.agents[idx]
                if other_agent != agent:
                    agent.trade(other_agent)
                    return
        else:
            # Fallback method if kdtree is not available
            for other_agent in world.agents:
                if other_agent != agent and self.is_nearby(agent, other_agent):
                    agent.trade(other_agent)
                    return

    def is_nearby(self, agent1: Agent, agent2: Agent, distance: int = 1) -> bool:
        return abs(agent1.x - agent2.x) <= distance and abs(agent1.y - agent2.y) <= distance

    def calculate_reward(self, agent: Agent) -> float:
        tile = self.world.grid[agent.y][agent.x]
        
        reward_components = {
            'health': 0.1 if agent.health > 50 else -0.1,
            'energy': 0.1 if agent.energy > 50 else -0.1,
            'explorer_bonus': 0.2 if agent.type == 'explorer' and tile.type not in ['grass', 'city'] else 0,
            'gatherer_bonus': 0.1 * agent.inventory if agent.type == 'gatherer' else 0,
            'crafter_bonus': agent.skills['crafting'] * 0.5 if agent.type == 'crafter' else 0,
            'city_trade': agent.inventory * 0.2 if agent.inventory > 0 and tile.type == 'city' else 0,
            'pollution_penalty': -tile.pollution * 0.1,
            'skills_bonus': sum(agent.skills.values()) * 0.1,
            'goal_progress': agent.goal_progress * 0.01
        }
        
        reward = sum(reward_components.values())
        
        if tile.type == 'city':
            agent.inventory = 0
        
        return reward

    def draw(self, screen: pygame.Surface) -> None:
        self.world.draw(screen, self.camera)
        tile_size = int(min(screen.get_width() // self.world.width, screen.get_height() // self.world.height) * self.zoom)
        for agent in self.agents:
            agent.draw(screen, tile_size, self.camera)

        self.draw_hud(screen)

    def draw_hud(self, screen: pygame.Surface) -> None:
        font = pygame.font.Font(None, 36)
        hud_info = [
            f"Step: {self.step_count}",
            f"Weather: {self.world.weather}",
            f"Agents: {len(self.agents)}"
        ]
        for i, text in enumerate(hud_info):
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10 + i * 40))
