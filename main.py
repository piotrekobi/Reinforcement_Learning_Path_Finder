import numpy as np
from q_learn import train_model, get_rewards, get_best_path, print_path

np.random.seed(2000)

lake = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

nrows = len(lake[0])
ncolumns = len(lake)
q_values = np.zeros((nrows, ncolumns, 4))
rewards = get_rewards(lake)

learning_rate = 0.9
discount_factor = 0.9
epsilon = 0.9
training_iterations = 1000

train_model(
    training_iterations,
    nrows,
    ncolumns,
    rewards,
    q_values,
    epsilon,
    discount_factor,
    learning_rate,
)
best_path, best_path_length = get_best_path(rewards, nrows, ncolumns, q_values)
print("Path found: ")
print_path(nrows, ncolumns, lake, best_path)
print(f"Length: {best_path_length}")
