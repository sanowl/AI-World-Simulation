# AI World Simulation V3

## Overview

This project is an  AI-driven world simulation that models a complex ecosystem of agents interacting with their environment. The simulation uses reinforcement learning techniques to allow agents to learn and adapt to their surroundings over time.

## Features

- Dynamic world generation with various terrain types (grass, water, mountain, forest, desert, city)
- Weather system affecting resource distribution and agent behavior
- Multiple agent types (explorer, gatherer, crafter) with unique behaviors and goals
- Deep Q-Network (DQN) based learning for agent decision making
- Visual representation using Pygame

## Project Structure

The project consists of several Python files:

- `main.py`: The entry point of the simulation, handling the game loop and user input
- `simulation.py`: Contains the `Simulation` class, which manages the overall simulation logic
- `world.py`: Defines the `World` class, responsible for world generation and management
- `agent.py`: Contains the `Agent` class, representing individual agents in the simulation
- `tile.py`: Defines the `Tile` class, representing individual tiles in the world
- `dqn_agent.py`: Implements the `DQNAgent` class for reinforcement learning

## Dependencies
:
This project requires the following Python libraries:

- Pygame
- NumPy
- PyTorch
- SciPy

You can install these dependencies using pip:

```
pip install pygame numpy torch scipy
```

## Running the Simulation

To run the simulation:

1. Ensure all dependencies are installed
2. Navigate to the project directory in your terminal
3. Run the following command:

```
python main.py
```

## Controls

- Arrow keys: Move the camera
- +/- keys: Zoom in/out
- Space: Pause/resume simulation

## Extending the Simulation

You can extend the simulation by:

- Adding new agent types in `agent.py`
- Implementing new terrain types in `tile.py`
- Modifying the reward function in `simulation.py` to encourage different behaviors
- Adjusting the DQN architecture in `dqn_agent.py` for different learning patterns

## Contributing

Contributions to improve the simulation are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).
