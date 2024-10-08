# Pinball Wizard

A reinforcement learning project to train an AI to play the classic 3D Pinball game for Windows - Space Cadet using Stable Baselines3 and TensorFlow.

## Table of Contents

1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact Information](#contact-information)

## Project Description

The Pinball Wizard AI project aims to create an AI agent that can learn to play the 3D Pinball game for Windows - Space Cadet. The project uses reinforcement learning algorithms implemented with Stable Baselines3 and logs training data using TensorBoard for analysis.

## Installation

To set up this project locally, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/MarvinWalls/PinballWizard.git
   cd PinballWizard
   ```

2. **Set up a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the AI agent, run the following command:

```
python main.py
```

### Game Setup

Ensure you have the 3D Pinball game for Windows - Space Cadet installed on your system. The game window should be visible and not minimized during training.

### TensorBoard

To visualize the training data, start TensorBoard:

```
tensorboard --logdir=tensorboard_logs
```

Then open your web browser and navigate to `http://localhost:6006`.

## Project Structure

Here's a brief overview of the project structure:

```
PinballWizard/
│
├── GameplayData/                  # Directory to store gameplay data
│
├── Screenshots/                   # Directory to store screenshots taken during gameplay
│
├── templates/                     # Directory containing templates for object detection
│
├── tensorboard_logs/              # Directory for TensorBoard logs
│
├── venv/                          # Virtual environment directory
│
├── frame_processor.py             # Script for processing game frames
├── game_control.py                # Script for controlling the game and extracting game data
├── keyboard_actions.py            # Script for performing keyboard actions in the game
├── main.py                        # Main script to start training
├── object_detection.py            # Script for object detection in game frames
├── pinball_env.py                 # Custom Gym environment for the pinball game
├── reward_system.py               # Script for calculating rewards based on game state
├── screen_capture.py              # Script for capturing game screenshots
├── tensorboard_callback.py        # Custom callback for logging data to TensorBoard
├── tensorboard_logger.py          # Logger script for TensorBoard metrics
└── requirements.txt               # List of dependencies
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Please ensure your code follows the project's coding standards and includes relevant tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information

For any questions or suggestions, please visit my website for contact information https://marvinwalls.github.io/my-portfolio/
```
