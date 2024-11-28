import pandas as pd
import numpy as np

class KFold:

    def __init__(self, embeddings=None, cluster=False):
        self.hr = []
        self.lr = []
        self.mv = MatrixVectorizer()
        if not cluster:
            for i in range(1, 4):
                self.hr.append(pd.read_csv(f"./Random CV/Fold{i}/hr_split_{i}.csv"))
                self.lr.append(pd.read_csv(f"./Random CV/Fold{i}/lr_split_{i}.csv"))
        else:
            for i in range(1, 4):
                letter = 'A' if i == 1 else 'B' if i == 2 else 'C'
                self.hr.append(pd.read_csv(f"./Cluster CV/Fold{i}/hr_cluster{letter}.csv"))
                self.lr.append(pd.read_csv(f"./Cluster CV/Fold{i}/hr_cluster{letter}.csv"))
        if embeddings != None:
            self.embeddings = []
            for i in range(1, 4):
                #load npy file
                self.embeddings.append(np.load(f"{embeddings}_fold_{i}.npy", allow_pickle=True))


    def obtain_folds(self, i, *, with_validation, return_embeddings=False):
        test_set = self.lr[i], self.hr[i]
        if return_embeddings:
            test_embeddings = self.embeddings[i]
        if i == 0:
            a, b = 1, 2
        elif i == 1:
            a, b = 0, 2
        else:
            a, b = 0, 1
        if not with_validation:
            train_set = pd.concat([self.lr[a], self.lr[b]]), pd.concat([self.hr[a], self.hr[b]])
            if return_embeddings:
                train_embeddings = np.concatenate([self.embeddings[a], self.embeddings[b]])
        else:
            val_set = pd.concat([self.lr[a].iloc[-20:], self.lr[b].iloc[-20:]]), pd.concat([self.hr[a].iloc[-20:], self.hr[b].iloc[-20:]])
            train_set = pd.concat([self.lr[a].iloc[:-20], self.lr[b].iloc[:-20]]), pd.concat([self.hr[a].iloc[:-20], self.hr[b].iloc[:-20]])
            if return_embeddings:
                val_embeddings = np.concatenate([self.embeddings[a][-20:], self.embeddings[b][-20:]])
                train_embeddings = np.concatenate([self.embeddings[a][:-20], self.embeddings[b][:-20]])
        if with_validation:
            print(train_set[0].shape, train_set[1].shape, test_set[0].shape, test_set[1].shape, val_set[0].shape, val_set[1].shape)
            if return_embeddings:
                return train_set[0], train_set[1], val_set[0], val_set[1], train_embeddings, val_embeddings
            else:
                return train_set[0], train_set[1], val_set[0], val_set[1]
        else:
            if return_embeddings:
                return train_set[0], train_set[1], test_set[0], test_set[1], train_embeddings, test_embeddings
            else:
                return train_set[0], train_set[1], test_set[0], test_set[1]

    def preprocessing(self):
        hr_trains = [None, None, None]
        lr_trains = [None, None, None]

        for i in range(3):
            self.lr[i] = self.lr[i].clip(lower=0).fillna(0)
            lr_trains[i] = self.lr[i].apply(lambda row: self.mv.anti_vectorize(row, 160), axis=1)

            self.hr[i] = self.hr[i].clip(lower=0).fillna(0)
            hr_trains[i] = self.hr[i].apply(lambda row: self.mv.anti_vectorize(row, 268), axis=1)

        self.lr = lr_trains
        self.hr = hr_trains


class MatrixVectorizer:
    def _init_(self):
        pass

    @staticmethod
    def vectorize(matrix, include_diagonal=False):
        # Determine the size of the matrix based on its first dimension
        matrix_size = matrix.shape[0]
        # Initialize an empty list to accumulate vector elements
        vector_elements = []

        # Iterate over columns and then rows to collect the relevant elements
        for col in range(matrix_size):
            for row in range(matrix_size):
                # Skip diagonal elements if not including them
                if row != col:  
                    if row < col:
                        # Collect upper triangle elements
                        vector_elements.append(matrix[row, col])
                    elif include_diagonal and row == col + 1:
                        # Optionally include the diagonal elements immediately below the diagonal
                        vector_elements.append(matrix[row, col])

        return np.array(vector_elements)

    @staticmethod
    def anti_vectorize(vector, matrix_size, include_diagonal=False):
        matrix = np.zeros((matrix_size, matrix_size))
        vector_idx = 0

        for col in range(matrix_size):
            for row in range(matrix_size):
                # Skip diagonal elements if not including them
                if row != col:  
                    if row < col:
                        # Reflect vector elements into the upper triangle and its mirror in the lower triangle
                        matrix[row, col] = vector[vector_idx]
                        matrix[col, row] = vector[vector_idx]
                        vector_idx += 1
                    elif include_diagonal and row == col + 1:
                        # Optionally fill the diagonal elements after completing each column
                        matrix[row, col] = vector[vector_idx]
                        matrix[col, row] = vector[vector_idx]
                        vector_idx += 1

        return matrix