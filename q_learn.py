import numpy as np
import random


def is_terminal(row, column, rewards):
    if rewards[row][column] == -1:
        return False
    return True


def next_move(current_row, current_column, nrows, ncolumns, q_values, epsilon):
    if np.random.random() < epsilon:
        action = np.argmax(q_values[current_row][current_column])
    else:
        action = np.random.randint(4)

    if action == 0:  # up
        current_row = max(current_row - 1, 0)
    elif action == 1:  # right
        current_column = min(current_column + 1, ncolumns - 1)
    elif action == 2:  # down
        current_row = min(current_row + 1, nrows - 1)
    elif action == 3:  # left
        current_column = max(current_column - 1, 0)
    return current_row, current_column, action


def get_rewards(map):
    nrows = len(map[0])
    ncolumns = len(map)
    rewards = np.zeros((nrows, ncolumns), dtype=int)
    rewards_dictionary = {"S": -1, "F": -1, "H": -100, "G": 100}
    for row in range(nrows):
        for column in range(ncolumns):
            rewards[row][column] = rewards_dictionary.get(map[row][column])
    return rewards


def train_model(
    iterations,
    nrows,
    ncolumns,
    rewards,
    q_values,
    epsilon,
    discount_factor,
    learning_rate,
):
    for _ in range(iterations):
        current_row, current_column = 0, 0

        while not is_terminal(current_row, current_column, rewards):
            prev_row, prev_column = current_row, current_column

            current_row, current_column, action = next_move(
                current_row, current_column, nrows, ncolumns, q_values, epsilon
            )
            reward = rewards[current_row, current_column]
            prev_point_q_value = q_values[prev_row, prev_column, action]
            temporal_difference = (
                reward
                + (discount_factor * np.max(q_values[current_row, current_column]))
                - prev_point_q_value
            )
            q_value = prev_point_q_value + learning_rate * temporal_difference
            q_values[prev_row, prev_column, action] = q_value


def get_best_path(rewards, nrows, ncolumns, q_values):
    row, column = 0, 0
    length = 0
    path = []
    path.append((row, column))
    while not is_terminal(row, column, rewards):
        row, column, _ = next_move(row, column, nrows, ncolumns, q_values, 1)
        path.append((row, column))
        length += 1
    return path, length


def print_path(nrows, ncolumns, map, path):
    for row in range(nrows):
        row_text = []
        for column in range(ncolumns):
            if (row, column) in path:
                row_text.append(map[row][column])
            else:
                row_text.append(" ")
        print("".join(row_text))
