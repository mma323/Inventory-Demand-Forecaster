import create_dataset
import decision_tree_regressor
import random_forest
import MLPRegressors
import matplotlib.pyplot as plt
import pandas as pd
from create_dataset import plot_data, create_dataset

def main():
    plt.show()

    data_2023 = pd.read_csv("data_2023.csv")
    data_2023 = create_dataset(data_2023, "Date", "Demand", 5)


    # The data from 2023 is missing values for all days after 25.08.2023    
    # Generate list of dates from 26.08.2023 to 31.12.2023 in the format dd.mm.yyyy
    # using pandas date_range function
    dates = pd.date_range(start="2023-08-26", end="2023-12-31").strftime("%d.%m.%Y").tolist()


    # Predict the demand for the missing days using an ensemble of MLP regressors
    ensemble = MLPRegressors.create_ensemble_of_mlp_regressors(10)
    ensemble = MLPRegressors.train_ensemble(ensemble, data_2023)
    predictions = []
    for i in range(len(dates)):
        data_points = data_2023[0][i]
        predictions.append(MLPRegressors.predict_ensemble(ensemble, data_points))
    
    plt.plot(
        dates, 
        predictions, 
        label="MLP predictions"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


