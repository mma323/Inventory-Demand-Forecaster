from sklearn.neural_network import MLPRegressor
from create_dataset import create_dataset, split
import pandas as pd
import matplotlib.pyplot as plt
import random as rdm

def create_ensemble_of_mlp_regressors(number_of_mlp_regressors):
    ensemble = []
    for i in range(number_of_mlp_regressors):
        regressor = MLPRegressor()
        ensemble.append(regressor)
    return ensemble
    

#Train the ensemble using bagging (comment)
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


#implement aggregation to combine the output of all MLP regressors to a single value
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
    ensemble = create_ensemble_of_mlp_regressors(10)
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
        label="MLP predictions"
    )
    plt.legend()

main()