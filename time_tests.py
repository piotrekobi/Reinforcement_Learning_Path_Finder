import numpy as np
import timeit
import csv
from q_learn import train_model, get_rewards


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
training_iterations = 1000


def time_model(learning_rate, discount_factor, epsilon):
    time = (
        timeit.timeit(
            lambda: train_model(
                training_iterations,
                nrows,
                ncolumns,
                rewards,
                q_values,
                epsilon,
                discount_factor,
                learning_rate,
            ),
            number=20,
        )
        / 20
    )
    return (
        round(time, 2),
        round(learning_rate, 1),
        round(discount_factor, 1),
        round(epsilon, 1),
    )


time_info = []
for learning_rate in np.linspace(0.8, 1, 3):
    for discount_factor in np.linspace(0.8, 1, 3):
        for epsilon in np.linspace(0.8, 0.9, 2):
            time_info.append(time_model(learning_rate, discount_factor, epsilon))

with open("times.csv", "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(
        [
            "Średni czas",
            "Współczynnik uczenia",
            "Stopa dyskontowa",
            "Wartość epsilon",
        ],
    )
    csv_writer.writerows(time_info)
    print("CSV created")
