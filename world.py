import random
from tile import Tile
from scipy.spatial import cKDTree

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
        tile_size = min(screen.get_width() // self.width, screen.get_height() // self.height)
        for y in range(self.height):
            for x in range(self.width):
                screen_x = x * tile_size - camera[0]
                screen_y = y * tile_size - camera[1]
                if 0 <= screen_x < screen.get_width() and 0 <= screen_y < screen.get_height():
                    self.grid[y][x].draw(screen, screen_x, screen_y, tile_size)

        if self.weather == 'rainy':
            for _ in range(100):
                rx = random.randint(0, screen.get_width())
                ry = random.randint(0, screen.get_height())
                pygame.draw.line(screen, (200, 200, 255), (rx, ry), (rx, ry + 10), 1)
        elif self.weather == 'stormy':
            for _ in range(5):
                rx = random.randint(0, screen.get_width())
                ry = random.randint(0, screen.get_height())
                pygame.draw.line(screen, (255, 255, 0), (rx, ry), (rx + random.randint(-20, 20), ry + random.randint(-20, 20)), 2)

    def update_kdtree(self):
        self.kdtree = cKDTree([(agent.x, agent.y) for agent in self.agents])
