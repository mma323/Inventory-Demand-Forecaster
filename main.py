import create_dataset
import decision_tree_regressor
import random_forest
import MLPRegressors
import matplotlib.pyplot as plt
import pandas as pd


def main():
    plt.show()

    window_size = 5
    data = pd.read_csv("data_2023.csv")
    training_data, testing_data = create_dataset.split(data, 0.8)

    training_dataset = create_dataset.create_dataset(
        training_data, "Date", "Demand", window_size
    )
    testing_dataset = create_dataset.create_dataset(
        testing_data, "Date", "Demand", window_size
    )


if __name__ == "__main__":
    main()


