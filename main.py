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

    # The 2023 data is missing the values after 25.08.2023.
    # Use MLP regressor ensemble to predict the missing values from
    # 26.08.2023 to 31.12.2023, save the predictions in a list.

    # Create ensemble of MLP regressors
    ensemble = MLPRegressors.create_ensemble_of_mlp_regressors(10)
    # Train ensemble
    ensemble = MLPRegressors.train_ensemble(ensemble, training_dataset)

    # Generate dates from 26.08.2023 to 31.12.2023
    start_date = pd.to_datetime("2023-08-26")
    end_date = pd.to_datetime("2023-12-31")
    date_range = pd.date_range(start_date, end_date)

    # Predict values for each date in the range
    predictions = []
    for date in date_range:
        data_points = testing_dataset[testing_dataset["Date"] <= date]["Demand"].tail(window_size).values
        prediction = MLPRegressors.predict_ensemble(ensemble, data_points)
        predictions.append(prediction)

    print(len(predictions))



if __name__ == "__main__":
    main()


