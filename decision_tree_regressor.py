from create_dataset import create_dataset, split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import random as rdm   

def train_regression_tree(dataset : tuple = None):
    """
        Trains a regression tree using the dataset provided.

        Parameters
        ----------
        dataset : tuple
            Tuple containing the data points, expected outputs and time values.
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
        regressor : DecisionTreeRegressor
            Trained regression tree.
        """
    if dataset is None:
        #random parameters
        regressor = DecisionTreeRegressor(
            random_state=0,
            criterion=rdm.choice(["poisson", "squared_error", "friedman_mse", "absolute_error"]),
            splitter=rdm.choice(["best", "random"]),
            max_depth=rdm.randint(1, 10),
            min_samples_split=rdm.randint(2, 10),
            min_samples_leaf=rdm.randint(1, 10),
            min_weight_fraction_leaf=rdm.random()*0.5,
            max_features=rdm.choice(["auto", "sqrt", "log2"]),
            max_leaf_nodes=rdm.randint(2, 10),
            min_impurity_decrease=rdm.random(),
            ccp_alpha=rdm.random()
        )
        return regressor
    
    data_points, expected_outputs, time = dataset
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(data_points, expected_outputs)

    return regressor


def main():
    window_size = 5
    data = pd.read_csv("data_2022.csv")
    training_data, testing_data = split(data, 0.8)
    training_dataset = create_dataset(
        training_data, "Date", "Demand", window_size
    )
    regressor = train_regression_tree(training_dataset)

    
    #Using the same sliding window approach as in (2.a), start from 01.01.2022 
    #and use the regression tree to predict the next value. Save each 
    #prediction in a list.
    predictions = []
    for i in range(len(testing_data) - window_size):
        data_points = testing_data["Demand"].iloc[i:i + window_size].values
        predictions.append(regressor.predict([data_points]))

    #Use matplotlib to plot the results alongside the original data.
    plt.plot(
        testing_data["Date"].iloc[window_size:], 
        predictions, 
        label="Decision tree predictions",
    )
    plt.plot(
        testing_data["Date"].iloc[window_size:], 
        testing_data["Demand"].iloc[window_size:], 
        label="Original data",
        linewidth=3
    )
    plt.legend()

main()