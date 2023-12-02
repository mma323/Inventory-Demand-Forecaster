import pandas as pd

def create_dataset(csv_path : str, column : str, window_size : int) -> tuple:
    """
        Reads data from a csv file and creates a dataset using the
        sliding window method.

        Parameters
        ----------
        csv_path : str
            Path to the csv file.
        column : str
            Name of the column in the csv file that will be used to create 
            the dataset.
        window_size : int
            Size of the window that will be used to create the dataset in
            the sliding window method.

        Returns
        -------
        data_points : list of lists
            List of lists containing the data points that are used as input
        expected_outputs : list
            List containing the next data point after the data points in
            each list of data_points.
        """
    data_frame = pd.read_csv(csv_path)
    data = data_frame[column].values

    data_points, expected_outputs = [], []

    for i in range(len(data) - window_size):
        data_points.append( data[i : i + window_size] )
        expected_outputs.append( data[i + window_size] )
    
    return data_points, expected_outputs



