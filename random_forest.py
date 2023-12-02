from decision_tree_regressor import train_regression_tree
from sklearn.tree import DecisionTreeRegressor
from create_dataset import create_dataset, split
import random as rdm
import pandas as pd
import matplotlib.pyplot as plt


#Create an ensemble of regression trees, each tree should be initialized with
#random parameters.
def create_ensemble(number_of_trees):
    ensemble = []
    for i in range(number_of_trees):
        regressor = train_regression_tree(None)
        ensemble.append(regressor)
    return ensemble


#implement bootstrapping and train each tree in the ensemble on a random
#part of the dataset.
def train_ensemble(ensemble, training_dataset):
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


#implement aggregation to combine the output of all your trees to a single value
def predict_ensemble(ensemble, data_points):
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

    #Using the same sliding window approach as in (2.a), start from 01.01.2022 
    #and use the regression tree to predict the next value. Save each 
    #prediction in a list.
    predictions = []
    for i in range(len(testing_data) - window_size):
        data_points = testing_data["Demand"].iloc[i:i + window_size].values
        predictions.append(predict_ensemble(ensemble, data_points))

    #Plot the predictions and the actual values in the same plot.
    plt.plot(
        testing_data["Date"].iloc[window_size:], 
        predictions, 
        label="Predictions"
    )
    plt.plot(
        testing_data["Date"].iloc[window_size:], 
        testing_data["Demand"].iloc[window_size:], 
        label="Original data"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()