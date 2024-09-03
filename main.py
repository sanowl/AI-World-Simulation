import pygame
from simulation import Simulation

pygame.init()

WIDTH, HEIGHT = 1600, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced AI World Simulation V3")

clock = pygame.time.Clock()

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

        screen.fill((0, 0, 0))
        simulation.draw(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
