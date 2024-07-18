import os
import json


def save_step_data(data, file_path):
    """
    Saves a single step of data to a file. The data should include information such as timestamp, action, reward,
    and the path to the screenshot of the game state at the time of the action.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Append the data as a JSON object to the specified file
    with open(file_path, 'a') as file:
        json.dump(data, file)
        file.write('\n')  # Add a newline to separate entries
def load_data_from_file(file_path):
    """
    Loads step data from a file. Returns a list of dictionaries, each representing a step in the game.
    """
    data = []
    if not os.path.exists(file_path):
        return data  # Return an empty list if the file doesn't exist

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line contains data
                entry = json.loads(line.strip())
                data.append(entry)
    return data

