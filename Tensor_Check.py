from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Correct path with a raw string
event_file_path = r'C:\Users\marvi\Pinball Wizard\logs\PPO_7\events.out.tfevents.1712094775.LAPTOP-0FE8U4UQ.26096.0'
accumulator = EventAccumulator(event_file_path)
accumulator.Reload()

# List all scalar tags
print("Available scalar tags:", accumulator.Tags()["scalars"])

# Replace 'reward' with the actual tag name found in the list
# For example, if 'episode_reward' is in the list of scalar tags:
# print(accumulator.Scalars('episode_reward'))