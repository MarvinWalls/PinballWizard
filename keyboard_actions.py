import pyautogui

# Define the actions as functions
def press_key(key):
    pyautogui.keyDown(key)
    print(f"Key down: {key}")

def release_key(key):
    pyautogui.keyUp(key)
    print(f"Key up: {key}")

def press_left_flipper():
    press_key('a')

def release_left_flipper():
    release_key('a')

def press_right_flipper():
    press_key('d')

def release_right_flipper():
    release_key('d')

def press_left_table_bump():
    press_key('x')

def release_left_table_bump():
    release_key('x')

def press_right_table_bump():
    press_key('.')

def release_right_table_bump():
    release_key('.')

def press_bottom_table_bump():
    press_key('Up')

def release_bottom_table_bump():
    release_key('Up')

def press_enter():
    pyautogui.press('enter')
    print("Pressed: enter")

def press_f2():
    pyautogui.press('f2')
    print("Pressed: f2")

def press_plunger():
    press_key('space')

def release_plunger():
    release_key('space')

def no_action():
    pass

# Updated perform_action function to map integer actions to action functions
def perform_action(action):
    actions = [
        no_action, press_left_flipper, release_left_flipper, press_right_flipper,
        release_right_flipper, press_plunger, release_plunger, press_left_table_bump,
        release_left_table_bump, press_right_table_bump, release_right_table_bump,
        press_bottom_table_bump, release_bottom_table_bump
    ]
    action_function = actions[action] if action < len(actions) else no_action
    action_function()
    print(f"Executed action: {action}")
