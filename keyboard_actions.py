import pyautogui

# Define the actions as functions
def press_key(key):
    pyautogui.keyDown(key)

def release_key(key):
    pyautogui.keyUp(key)

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

def press_f2():
    pyautogui.press('f2')

def press_plunger():
    press_key('space')

def release_plunger():
    release_key('space')

def no_action():
    pass

# Updated perform_action function to map integer actions to action functions
def perform_action(action):
    actions = {
        0: no_action,
        1: press_left_flipper,
        2: release_left_flipper,
        3: press_right_flipper,
        4: release_right_flipper,
        5: press_plunger,
        6: release_plunger,
        7:press_left_table_bump,
        8:release_left_table_bump,
        9:press_right_table_bump,
        10:release_right_table_bump,
        11:press_bottom_table_bump,
        12:release_bottom_table_bump
    }

    action_function = actions.get(action)
    if action_function:
        action_function()
    else:
        print(f"Unknown action: {action}")