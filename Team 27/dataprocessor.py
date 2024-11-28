import numpy as np
from MatrixVectorizer import MatrixVectorizer as vectorizer

class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def load_data(file_path: str) -> np.ndarray:
        """
        Loads provided csv data into numpy array.

        *The first row is used as the columns indexes, which will be discarded.*

        :param file_path: relative path to csv file
        :return: 2d array, each row is a vectorized data entry
        """
        data = np.loadtxt(file_path, delimiter=',')[1:]
        print(f"Loaded file {file_path} with shape {data.shape}")
        return data

    @staticmethod
    def load_data_matrices(file_path: str, dim: int) -> np.ndarray:
        """
        Loads provided csv data into numpy array of matrices, in the anti vectorized form.

        *Need to manually specify resolution of the image.*

        :param file_path: relative path to csv file
        :param dim: given the loaded image is dim x dim
        :return: 3d array, each 'row' is an anti-vectorized data matrix
        """
        data = np.loadtxt(file_path, delimiter=',')[1:]
        print(f"Loaded file {file_path} with shape {data.shape}")

        num_samples = data.shape[0]
        data_anti_vectorized = np.zeros([num_samples, dim, dim])
        for i in range(num_samples):
            data_anti_vectorized[i] = vectorizer.anti_vectorize(data[i], dim)
        print(f"Anti-vectorized to {data_anti_vectorized.shape}")
        return data_anti_vectorized

    @staticmethod
    def save_data(data: np.ndarray, file_path: str):
        """
        I use this to temporarily store results, so I won't lost them

        *A dummy first row will be added, so it keeps consistent through saves and loads*
        """
        dummy_heading = np.zeros([data.shape[1]])

        data = np.vstack([dummy_heading, data])
        np.savetxt(file_path, data, delimiter=",")
        print(f"Saved file {file_path}")

    @staticmethod
    def save_kaggle_csv(data: np.ndarray, file_path: str):
        """
        Save predicted HR matrices to kaggle-friendly csv file, as specified.

        :param data: (112, 35778) numpy array of predicted HR
        :param file_path: relative path to csv file
        """
        # Check the dimensions
        if data.shape != (112, 268, 268):
            print(f"Prediction Matrix shape is {data.shape}, not (112, 268, 268)!")
            # Just a warning, still proceed

        number_of_rows = data.shape[0]
        # Vectorize data
        data_vec = np.zeros([number_of_rows, 35778])
        for i in range(number_of_rows):
            data_vec[i] = vectorizer.vectorize(data[i])

        # Construct data entries
        melted_data = data_vec.flatten()
        dummy_index = np.arange(1, len(melted_data) + 1, dtype=np.int32)
        data_t = melted_data.reshape(-1, 1)
        data_combined = np.column_stack((dummy_index, data_t))

        # Construct csv file
        np.savetxt(file_path, data_combined, delimiter=',',
                   header="ID,Predicted", fmt=['%d', '%.16g'], comments='')

        print(f".csv file '{file_path}' created.")

    @staticmethod
    def fix_kaggle_submission(file_path: str):
        """
        A single use fix. Hopefully we won't need it later.
        """
        data = np.loadtxt(file_path, skiprows=1, delimiter=',')
        print(data[3378])
        np.savetxt('data/test_kaggle.csv', data, delimiter=',',
                   header="ID,Predicted", fmt=['%d', '%.16g'], comments='')
