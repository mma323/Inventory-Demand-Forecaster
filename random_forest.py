from decision_tree_regressor import train_regression_tree
from sklearn.tree import DecisionTreeRegressor
from create_dataset import create_dataset, split
import random as rdm
import pandas as pd
import matplotlib.pyplot as plt


def create_ensemble(number_of_trees : int) -> list:
    """
        Creates an ensemble of regression trees.

        Parameters
        ----------
        number_of_trees : int
            Number of regression trees in the ensemble.

        Returns
        -------
        list
            Ensemble of regression trees.
    """
    ensemble = []
    for i in range(number_of_trees):
        regressor = train_regression_tree(None)
        ensemble.append(regressor)

    return ensemble


def train_ensemble(ensemble : list, training_dataset : tuple) -> list:
    """
        Trains an ensemble of regression trees.

        Parameters
        ----------
        ensemble : list
            Ensemble of regression trees.
        training_dataset : tuple
            Containing the data points, expected outputs and time values.
            data_points : list of lists
                List of lists containing the data points that are used as input
            expected_outputs : list
                List containing the next data point after the data points in
                each list of data_points.
            time : list
                List containing the time values for each data point, is not
                used in this function.

        Returns
        -------
        list
            Trained ensemble of regression trees.
    """
    data_points, expected_outputs, time = training_dataset

    for regressor in ensemble:
        data_points_selection = []
        expected_outputs_selection = []

        for i in range(len(data_points)):
            index = rdm.randint(0, len(data_points) - 1)
            data_points_selection.append(data_points[index])
            expected_outputs_selection.append(expected_outputs[index])

        regressor.fit(data_points_selection, expected_outputs_selection)

    return ensemble


def predict_ensemble(ensemble: list, data_points: list) -> float:
    """
        Predicts the next data point using an ensemble of regression trees.

        Parameters
        ----------
        ensemble : list
            Ensemble of regression trees.
        data_points : list
            List containing the data points that are used as input.

        Returns
        -------
        float
            Predicted value.
    """
    predictions = []
    for regressor in ensemble:
        predictions.append(regressor.predict([data_points]))

    return sum(predictions) / len(predictions)


def main():
    window_size = 5
    data = pd.read_csv("data_2022.csv")

    training_data, testing_data = split(data, 0.8)

    training_dataset = create_dataset(
        training_data, "Date", "Demand", window_size
    )

    ensemble = create_ensemble(10)
    ensemble = train_ensemble(ensemble, training_dataset)

    predictions = []
    for i in range(len(testing_data) - window_size):
        data_points = testing_data["Demand"].iloc[i:i + window_size].values
        predictions.append(predict_ensemble(ensemble, data_points))

    plt.plot(
        testing_data["Date"].iloc[window_size:], 
        predictions, 
        label="Random forest predictions"
    )
    plt.legend()


main()