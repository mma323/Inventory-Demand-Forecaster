import pandas as pd
import matplotlib.pyplot as plt


def plot_data(csv_path : str, data_column : str, time_column : str) -> None:
    """
        Plots the data from a csv file to see what it looks like in order to 
        determine window size for sliding window.

        Parameters
        ----------
        csv_path : str
            Path to the csv file containing the data.
        data_column : str
            Name of the column in the csv file that contains the data.
        time_column : str
            Name of the column in the csv file that contains the time values.
        """
    data_frame = pd.read_csv(csv_path)
    data = data_frame[data_column].values
    time = data_frame[time_column].values

    plt.plot(time, data)
    plt.show()


def split(data : pd.DataFrame, training_fraction : float) -> tuple:
    """
        Splits the data into training and testing data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be split.
        training_fraction : float
            Fraction of the data that should be used for training.

        Returns
        -------
        training_data : pd.DataFrame
            Training data.
        testing_data : pd.DataFrame
            Testing data.
        """
    split_index = int(len(data) * training_fraction)
    training_data = data[:split_index]
    testing_data = data[split_index:].reset_index(drop=True)
    return training_data, testing_data




def create_dataset(
        data : pd.DataFrame, 
        time_column : str, 
        data_column : str, 
        window_size : int
) -> tuple:
    """
        Creates a dataset from the data provided using a sliding window.

        Parameters
        ----------
        data : pd.DataFrame
            Data to create the dataset from.
        time_column : str
            Name of the column in the csv file that contains the time values.
        data_column : str
            Name of the column in the csv file that contains the data.
        window_size : int
            Size of the sliding window.

        Returns
        -------
        dataset : tuple
            Tuple containing the data points, expected outputs and time values.
            data_points : list of lists
                List of lists containing the data points that are used as input
            expected_outputs : list
                List containing the next data point after the data points in
                each list of data_points.
            time : list
                List containing the time values for each data point.
        """
    data_points = []
    expected_outputs = []
    time = []

    for i in range(len(data) - window_size):
        data_points.append(data[data_column][i:i + window_size].values)
        expected_outputs.append(data[data_column][i + window_size])
        time.append(data[time_column][i + window_size])

    return data_points, expected_outputs, time


def main():
    #For testing purposes
    file = "data_2022.csv"
    data_column = "Demand"
    time_column = "Date"
    window_size = 30

    plot_data(file, data_column, time_column)

    data_frame = pd.read_csv(file)
    #shuffle data
    training_data, testing_data = split(data_frame, 0.8)
   
    print(testing_data)



if __name__ == "__main__":
    main()