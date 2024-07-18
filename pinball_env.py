import gym
from gym import spaces
import numpy as np
from game_control import GameControl
from reward_system import RewardSystem


class PinballEnv(gym.Env):
    """Custom Environment for Pinball Wizard that follows gym interface."""
    metadata = {'render.modes': ['human']}

    def __init__(self, window_title, templates_directory, screenshot_dir):
        super(PinballEnv, self).__init__()
        self.action_space = spaces.Discrete(5)  # Adjust as necessary
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

        # Initialize GameControl with correct parameters
        self.game_control = GameControl(window_title, templates_directory, screenshot_dir)

        # Initialize RewardSystem
        self.reward_system = RewardSystem()
        self.reward = 0

    def step(self, action):
        """
        Executes the given action in the environment.
        Returns the next state, reward, done flag, and optional info.
        """
        # Execute the action and unpack the returned values from perform_action method.
        processed_frame, reward, done = self.game_control.perform_action(action)

        # Validate the processed_frame is not None and is the expected type before further processing.
        if processed_frame is None or not isinstance(processed_frame, np.ndarray):
            print("Error: Processed frame is None or not a valid image array.")
            return np.zeros((480, 640, 3)), 0, True, {}  # Return a default state and indicate the episode is done.

        # Ensure the processed_frame has the correct shape with _preprocess_state.
        processed_frame = self._preprocess_state(processed_frame)

        # Here you would extract the current score and ball count from the processed frame.
        # Placeholder methods `extract_score` and `extract_ball_count` are assumed to be correctly defined in GameControl.

    def reset\
                    (self):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        initial_state = self.game_control.reset_game()
        initial_state = self._preprocess_state(initial_state)
        self.reward_system.reset()
        self.reward = 0

        return initial_state

    def _preprocess_state(self, state):
        """
        Preprocesses the state to ensure it has the correct shape (480, 640, 3).
        """
        if state.shape != (480, 640, 3):
            state = np.transpose(state, (1, 0, 2))  # Swap height and width dimensions
        return state

    def render(self, mode='human', close=False):
        """
        Render the environment. This method is optional and may not be necessary for your setup.
        """
        pass

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass
