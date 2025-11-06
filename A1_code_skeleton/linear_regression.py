import numpy as np
from numpy.linalg import pinv 

def univariate_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param x: 1D array that represents the feature vector
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector theta = (b, w)
    :return: a scalar that represents the loss \mathcal{L}_U(theta)
    """
    # TODO: Implement the univariate loss \mathcal{L}_U(theta) (as specified in Equation 1)
    w = theta[1]
    b = theta[0]
    y_pred = b + w*x
    mse_loss = np.mean((y_pred-y)**2)
    return mse_loss


def fit_univariate_lin_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_U(theta)
    """

    N = x.size
    assert N > 1, "There must be at least 2 points given!"
    # TODO: Implement the 1D case of linear regression from the assigment sheet (see also slides from practicals)
    w = None
    b = None
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    w_star = np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
    b_star = y_mean-(w_star*x_mean)
    w = w_star
    b = b_star
    return np.array([b, w])


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: a scalar that represents the Pearson correlation coefficient between x and y
    """
    # TODO: Implement Pearson correlation coefficient, as shown in Equation 3 (Task 1.1.1).
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    pearson_r = np.sum((x-x_mean)*(y-y_mean))/(np.sqrt(np.sum((x-x_mean)**2))*np.sqrt(np.sum((y-y_mean)**2)))
    return pearson_r


def compute_design_matrix(data: np.ndarray) -> np.ndarray:
    """
    :param data: 2D array of shape (N, D) that represents the data matrix
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the design matrix for multiple linear regression (Task 1.2.2)
    N, D = data.shape
    design_matrix = np.empty((N, D+1), dtype=float)
    design_matrix[:, 0] = 1
    design_matrix[:, 1:] = data
    return design_matrix


def multiple_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector
    :return: a scalar that represents the loss \mathcal{L}_M(theta)
    """
    # TODO: Implement the multiple regression loss \mathcal{L}_M(theta) (as specified in Equation 5)
    y_pred = X @ theta
    mean_squared_error = np.mean((y_pred-y)**2)
    return mean_squared_error


def fit_multiple_lin_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_M(theta)
    """
    from numpy.linalg import pinv

    # TODO: Implement the solution to multivariate linear regression. 
    # Note: Use the pinv function for the Moore-Penrose pseudoinverse!
    theta = pinv(X) @ y
    return theta


def compute_polynomial_design_matrix(x: np.ndarray, K: int) -> np.ndarray:
    """
    :param x: 1D array that represents the feature vector
    :param K: the degree of the polynomial
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the polynomial design matrix (Task 1.3.2)
    N = x.shape
    polynomial_design_matrix = np.empty((N[0], K+1), dtype=float)
    polynomial_design_matrix[:, 0] = 1
    for i in range(1, K+1):
        polynomial_design_matrix[:, i] = (x)**i
    return polynomial_design_matrix