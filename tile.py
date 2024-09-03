import pygame

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
        }.get(self.type, (255, 255, 255))
        
        pollution_factor = min(self.pollution / 100, 1)
        color = tuple(int(c * (1 - pollution_factor) + 100 * pollution_factor) for c in color)
        
        pygame.draw.rect(screen, color, (x, y, size, size))
        if self.resource > 0:
            pygame.draw.circle(screen, (255, 255, 0), (x + size // 2, y + size // 2), size // 4)
