import pyautogui
import util
import numpy as np
from pynput.mouse import Button, Controller
keyboard = Controller()

screen_width, screen_height = pyautogui.size()
middle_mouse_toggle = False
shift_toggle = False

THR = 0.1

def detect_gestures(landmarks_list, results):
    text = "Neutral"
    
    # Two Hand
#     if len(landmarks_list) >= 21 & (len(results.multi_handedness) == 2):

#         # PANNING
#         if keymouse.is_pan_condition(landmarks_list) & (results.multi_handedness.classification[0].label[0:] == "Left"):
#             text = "Panning"
        
#         return text
    

    # One Hand
    if len(landmarks_list) >= 21 & (len(results.multi_handedness) == 1):

        # MOVE
        if is_move_condition(landmarks_list):
            text = "Move"

        # SCROLL
        elif is_scroll_condition(landmarks_list):
            text = "Scroll"

        # PANNING
        elif is_pan_condition(landmarks_list):
            text = "Panning"

        # TILTING
        elif is_tilt_condition(landmarks_list):
            text = "Tilting"

    return text
    

def pan_mode(index_finger_tip, results, prev_results):
    if index_finger_tip is not None:
        mouse_move(results, prev_results, True, False)

def tilt_mode(index_finger_tip, results, prev_results):
    if index_finger_tip is not None:
        mouse_move(results, prev_results, False, True)

# Move Condition
# L + closed third, fourth, fifth finger
def is_move_condition(landmarks_list):
    middle_down = util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) <= 65
    ring_down = util.get_angle(landmarks_list[13], landmarks_list[14], landmarks_list[16]) <= 65
    pinky_down = util.get_angle(landmarks_list[17], landmarks_list[18], landmarks_list[20]) <= 65
    index_up = (util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) <= 200) & ((util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) >= 160))
    thumb_up = (util.get_angle(landmarks_list[1], landmarks_list[2], landmarks_list[4]) <= 200) & ((util.get_angle(landmarks_list[1], landmarks_list[2], landmarks_list[4]) >= 160))
    l_shape = util.calculate_distance([landmarks_list[4], landmarks_list[8]]) >= 90
    
    return ((middle_down) & (ring_down) & (pinky_down) & (index_up) & (thumb_up) & (l_shape))

# Scroll Condition
# Star Trek?
def is_scroll_condition(landmarks_list):
    pinky_up = (util.get_angle(landmarks_list[17], landmarks_list[18], landmarks_list[20]) <= 200) & ((util.get_angle(landmarks_list[17], landmarks_list[18], landmarks_list[20]) >= 160))
    ring_up = (util.get_angle(landmarks_list[13], landmarks_list[14], landmarks_list[16]) <= 200) & ((util.get_angle(landmarks_list[13], landmarks_list[14], landmarks_list[16]) >= 160))
    middle_up = (util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) <= 200) & ((util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) >= 160))
    index_up = (util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) <= 200) & ((util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) >= 160))
    thumb_up = (util.get_angle(landmarks_list[1], landmarks_list[2], landmarks_list[4]) <= 200) & ((util.get_angle(landmarks_list[1], landmarks_list[2], landmarks_list[4]) >= 160))
    index_mid_dis = util.calculate_distance([landmarks_list[8], landmarks_list[12]]) <= 35
    ring_pinky_dis = util.calculate_distance([landmarks_list[16], landmarks_list[20]]) <= 30

    return ((pinky_up) & (ring_up) & (middle_up) & (index_up) & (thumb_up) & (index_mid_dis) & (ring_pinky_dis))

# Pan Condition
# L w/Third
def is_pan_condition(landmarks_list):
    middle_up = (util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) <= 200) & ((util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) >= 160))
    ring_down = util.get_angle(landmarks_list[13], landmarks_list[14], landmarks_list[16]) <= 65
    pinky_down = util.get_angle(landmarks_list[17], landmarks_list[18], landmarks_list[20]) <= 65
    index_up = (util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) <= 200) & ((util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) >= 160))
    thumb_up = (util.get_angle(landmarks_list[1], landmarks_list[2], landmarks_list[4]) <= 200) & ((util.get_angle(landmarks_list[1], landmarks_list[2], landmarks_list[4]) >= 160))
    l_shape = util.calculate_distance([landmarks_list[4], landmarks_list[8]]) >= 90
    index_mid_dis = util.calculate_distance([landmarks_list[8], landmarks_list[12]]) <= 35

    
    return ((middle_up) & (ring_down) & (pinky_down) & (index_up) & (thumb_up) & (l_shape) & (index_mid_dis))

# Tilt Condition
# Thumb Closed, Four Fingers Up
def is_tilt_condition(landmarks_list):
    index_up = (util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) <= 200) & ((util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) >= 160))
    middle_up = (util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) <= 200) & ((util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) >= 160))
    ring_up = (util.get_angle(landmarks_list[13], landmarks_list[14], landmarks_list[16]) <= 200) & ((util.get_angle(landmarks_list[13], landmarks_list[14], landmarks_list[16]) >= 160))
    pinky_up = (util.get_angle(landmarks_list[17], landmarks_list[18], landmarks_list[20]) <= 200) & ((util.get_angle(landmarks_list[17], landmarks_list[18], landmarks_list[20]) >= 160))
    thumb_slot = util.calculate_distance([landmarks_list[4], landmarks_list[5]]) <= 20

    index_mid_dis = util.calculate_distance([landmarks_list[8], landmarks_list[12]]) <= 35
    mid_ring_dis = util.calculate_distance([landmarks_list[12], landmarks_list[16]]) <= 40
    ring_pink_dis = util.calculate_distance([landmarks_list[16], landmarks_list[20]]) <= 50

    return ((index_up) & (middle_up) & (ring_up) & (pinky_up) & (thumb_slot) & (index_mid_dis) & (mid_ring_dis) & (ring_pink_dis))

# From albpurpura
def get_last_k_valid_reading(prev_results, k):
    valid_readings = []
    for i in range(1, len(prev_results)):
        prev_landmarks = get_hand_landmarks(prev_results[-i])
        if prev_landmarks is not None:
            valid_readings.append(prev_landmarks)
            if len(valid_readings) == k:
                return valid_readings
            # return prev_landmarks
    return valid_readings

def mouse_move(results, prev_results, middle_mouse_toggle, shift_key_toggle):
    curr_landmarks = get_hand_landmarks(results)
    prev_landmarks = get_last_k_valid_reading(prev_results, 5)[0]
    second_prev_landmarks = get_last_k_valid_reading(prev_results, 5)[-1]
    
    if curr_landmarks is None or prev_landmarks is None:
        return
    curr_x_coords = np.array([curr_landmarks.landmark[i].x for i in range(len(curr_landmarks.landmark))])[8]
    curr_y_coords = np.array([curr_landmarks.landmark[i].y for i in range(len(curr_landmarks.landmark))])[8]

    prev_x_coords = np.array([prev_landmarks.landmark[i].x for i in range(len(prev_landmarks.landmark))])[8]
    prev_y_coords = np.array([prev_landmarks.landmark[i].y for i in range(len(prev_landmarks.landmark))])[8]
    diff = (curr_x_coords - prev_x_coords, curr_y_coords - prev_y_coords)

    sec_prev_x_coords = np.array([second_prev_landmarks.landmark[i].x for i in range(len(second_prev_landmarks.landmark))])[8]
    sec_prev_y_coords = np.array([second_prev_landmarks.landmark[i].y for i in range(len(second_prev_landmarks.landmark))])[8]

    control_diff = (curr_x_coords - sec_prev_x_coords, curr_y_coords - sec_prev_y_coords)
    # print(control_diff)
    if np.absolute(control_diff[0]) >= 0.003 or np.absolute(control_diff[1]) >= 0.003:
        if middle_mouse_toggle:
            pyautogui.dragRel(int(diff[0] * screen_width * 3), int(diff[1] * screen_height * 3), duration=0.1, button=('middle'))
        elif shift_key_toggle:
            with pyautogui.hold('shift'):
                pyautogui.dragRel(int(diff[0] * screen_width * 3), int(diff[1] * screen_height * 3), duration=0.1, button='middle')
        else:
            pyautogui.moveRel(int(diff[0] * screen_width * 3), int(diff[1] * screen_height * 3), duration=0.1)

def get_hand_landmarks(results):
    if results.multi_handedness is None:
        return None
    multi_hand_landmarks = results.multi_hand_landmarks
    multi_handedness = results.multi_handedness

    hands_labels = [item.classification[0].label for item in multi_handedness if item.classification[0].label]

    if len(hands_labels) == 1:
        hands_index = 0
    else:
        if 'Right' in hands_labels and len(set(hands_labels)) == 2:
            hands_index = hands_labels.index('Right')
        else:
            return None
    hand_landmarks = multi_hand_landmarks[hands_index]
    return hand_landmarks

def scroll(results, prev_results):
    curr_landmarks = get_hand_landmarks(results)
    prev_landmarks = get_last_k_valid_reading(prev_results, 5)[0]

    if curr_landmarks is None or prev_landmarks is None:
        return

    curr_y_coords = np.array([curr_landmarks.landmark[i].y for i in range(len(curr_landmarks.landmark))])[8]
    prev_y_coords = np.array([prev_landmarks.landmark[i].y for i in range(len(prev_landmarks.landmark))])[8]
    diff = curr_y_coords - prev_y_coords
    # print(diff)
    if np.absolute(diff) >= 0.01:
        pyautogui.scroll(100 * diff)
        print('scrolled')