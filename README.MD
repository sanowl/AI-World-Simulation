# AI World Simulation

## Overview
This project is a simple AI-driven world simulation that models agents interacting with their environment. The simulation uses basic reinforcement learning to allow agents to adapt to their surroundings.

## Features
- Basic world generation with different terrain types
- Simple weather system
- Multiple agent types with different behaviors
- Rudimentary learning mechanism for agents
- Visual representation using Pygame

## Project Structure
The project consists of several Python files:
- `main.py`: The entry point of the simulation
- `simulation.py`: Contains the `Simulation` class for overall simulation logic
- `world.py`: Defines the `World` class for world management
- `agent.py`: Contains the `Agent` class for individual agents
- `tile.py`: Defines the `Tile` class for world tiles
- `dqn_agent.py`: Implements a basic learning mechanism for agents

## Dependencies
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
1. Install the dependencies
2. Navigate to the project directory
3. Run:
```
python main.py
```

## Controls
- Arrow keys: Move the camera
- +/- keys: Zoom in/out
- Space: Pause/resume simulation

## Extending the Simulation
You can extend the simulation by mo`difying the existing files to add new features or behaviors.

## Contributing
Contributions to improve the simulation are welcome. Feel free to submit pull requests or open issues for bugs and feature requests.


