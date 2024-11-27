import numpy as np


def anti_vectorize(vector, matrix_size, include_diagonal=False):
    """
    Reconstructs a matrix from its vector form, filling it vertically.
    
    The method fills the matrix by reflecting vector elements into the upper triangle
    and optionally including the diagonal elements based on the include_diagonal flag.
    
    Parameters:
    - vector (numpy.ndarray): The vector to be transformed into a matrix.
    - matrix_size (int): The size of the square matrix to be reconstructed.
    - include_diagonal (bool, optional): Flag to include diagonal elements in the reconstruction.
        Defaults to False.
    
    Returns:
    - numpy.ndarray: The reconstructed square matrix.
    """
    # Initialize a square matrix of zeros with the specified size
    matrix = np.zeros((matrix_size, matrix_size))

    # Index to keep track of the current position in the vector
    vector_idx = 0

    # Fill the matrix by iterating over columns and then rows
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


def anti_vectorize_optimized(vector, matrix_size, include_diagonal=False):
    """
    Optimized version of the anti_vectorize function using NumPy's advanced indexing
    and vectorization for faster execution.
    """
    # Calculate the number of elements in the upper triangle (excluding the diagonal)
    num_upper = matrix_size * (matrix_size - 1) // 2
    # If including the diagonal, adjust the number of elements to include it
    if include_diagonal:
        num_upper += matrix_size
    # Ensure the vector has the correct length
    assert len(vector) == num_upper, "Vector size does not match the specified matrix size with the given include_diagonal setting."

    # Initialize a square matrix of zeros with the specified size
    matrix = np.zeros((matrix_size, matrix_size))

    # Create indices for the upper triangle (or including the diagonal if specified)
    row_indices, col_indices = np.triu_indices(matrix_size, k=0 if include_diagonal else 1)
    vector = vector.detach().cpu().numpy()

    # Fill the upper triangle (and optionally the diagonal) using the vector
    matrix[row_indices, col_indices] = vector

    # Mirror the upper triangle to the lower triangle, excluding the diagonal
    if include_diagonal:
        matrix += matrix.T - np.diag(np.diag(matrix))
    else:
        matrix += matrix.T

    return matrix


def vectorize(matrix, include_diagonal=False):
        """
        Converts a matrix into a vector by vertically extracting elements.
        
        This method traverses the matrix column by column, collecting elements from the
        upper triangle, and optionally includes the diagonal elements immediately below
        the main diagonal based on the include_diagonal flag.
        
        Parameters:
        - matrix (numpy.ndarray): The matrix to be vectorized.
        - include_diagonal (bool, optional): Flag to include diagonal elements in the vectorization.
          Defaults to False.
        
        Returns:
        - numpy.ndarray: The vectorized form of the matrix.
        """
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