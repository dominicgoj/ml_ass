import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

from plot_utils import (plot_scatterplot_and_line, plot_scatterplot_and_polynomial, 
                        plot_logistic_regression, plot_datapoints, plot_3d_surface, plot_2d_contour,
                        plot_function_over_iterations, pairplot)
from linear_regression import (fit_univariate_lin_model, 
                               fit_multiple_lin_model, 
                               univariate_loss, multiple_loss,
                               calculate_pearson_correlation,
                               compute_design_matrix,
                               compute_polynomial_design_matrix)
from logistic_regression import (create_design_matrix_dataset_1,
                                 create_design_matrix_dataset_2,
                                 create_design_matrix_dataset_3,
                                 logistic_regression_params_sklearn)
from gradient_descent import rastrigin, gradient_rastrigin, gradient_descent


def task_1(use_linalg_formulation=False):
    print('---- Task 1 ----')
    column_to_id = {"hours_sleep": 0, "hours_work": 1,
                    "avg_pulse": 2, "max_pulse": 3, 
                    "duration": 4, "exercise_intensity": 5, 
                    "fitness_level": 6, "calories": 7}

    # After loading the data, you can for example access it like this: 
    # `smartwatch_data[:, column_to_id['hours_sleep']]`
    smartwatch_data = np.load('data/smartwatch_data.npy') ##  Load the smartwatch data from a .npy file 
    
    #pairplot(smartwatch_data, list(column_to_id.keys())) ##  Plot pairplot to visualize relationships between features
    chosen_pairs_linearly_dependent = [
        ['duration', 'fitness_level'],
        ['fitness_level', 'calories'],
        ['duration', 'calories']
    ]
    chosen_pairs_linearly_independent = [
        ['hours_sleep', 'avg_pulse'],
        ['fitness_level', 'max_pulse'],
        ['max_pulse', 'duration']
    ]

    all_pairs = []
    for column_1 in column_to_id:
        for column_2 in column_to_id:
            if column_1 != column_2 and not [column_1, column_2] in all_pairs and not [column_2, column_1] in all_pairs:
                pair = [column_1, column_2]
                all_pairs.append(pair)
            
    

    # TODO: Implement Task 1.1.1: Find 3 pairs of features that have a linear relationship.
    # For each pair, fit a univariate linear regression model: If ``use_linalg_formulation`` is False,
    # call `fit_univariate_lin_model`, otherwise use the linalg formulation in `fit_multiple_lin_model` (Task 1.2.2).
    # For each pair, also calculate and report the Pearson correlation coefficient, the theta vector you found, 
    # the MSE, and plot the data points together with the linear function.
    # Repeat the process for 3 pairs of features that do not have a meaningful linear relationship.
    #fulldata = []
    if not use_linalg_formulation:
        for pair in chosen_pairs_linearly_dependent+chosen_pairs_linearly_independent:
            data = smartwatch_data.copy()
            X = data[:, column_to_id[pair[0]]]
            y = data[:, column_to_id[pair[1]]]
            theta = fit_univariate_lin_model(x=X, y=y)
            mse_loss = univariate_loss(x=X, y=y, theta=theta)
            pearson = calculate_pearson_correlation(x=X, y=y)
            xlabel = pair[0].replace("_", " ").capitalize()
            ylabel = pair[1].replace("_", " ").capitalize()
            plot_title = f"{xlabel} vs. {ylabel}"
            plot_filename = f"uni_{pair[0]}_vs_{pair[1]}"
            plot_scatterplot_and_line(
                x=X,
                y=y,
                theta=theta,
                xlabel=pair[0],
                ylabel=pair[1],
                title=plot_title,
                figname=plot_filename
            )
            data = {
                'x': pair[0],
                'y': pair[1],
                'mse': round(mse_loss, 2),
                'w': round(theta[1], 2),
                'b': round(theta[0], 2),
                'pearson': round(pearson, 2)
            }
            #fulldata.append(data)
            print(f"Chosen pair: {pair[0]} vs {pair[1]}")
            print(f"MSE: {round(mse_loss,2)}, Theta: b:{round(theta[0],2)}, w:{round(theta[1],2)}; Pearson: {round(pearson, 2)}")
            print("--------")
    # TODO: Implement Task 1.2.2: Multiple linear regression
    # Select two additional features, compute the design matrix, and fit the multiple linear regression model.
    # Report the MSE and the theta vector.
    elif use_linalg_formulation:
        chosen_x = ['duration', 'fitness_level', 'exercise_intensity']
        chosen_y = 'calories'
        cols = [column_to_id[x] for x in chosen_x]
        data = smartwatch_data[:, cols]
        design_matrix_x = compute_design_matrix(data=data)
        y = smartwatch_data[:, column_to_id[chosen_y]]
        theta_star = fit_multiple_lin_model(X=design_matrix_x, y=y)
        mse_loss = multiple_loss(X=design_matrix_x, y=y, theta=theta_star)
        print(f"{chosen_x} vs {chosen_y}")
        print(f"MSE: {round(mse_loss, 2)}, Theta: {theta_star}")
    

    # TODO: Implement Task 1.3.1: Polynomial regression
    # For the feature-target pair of choice, compute the polynomial design matrix with an appropriate degree K, 
    # fit the model, and plot the data points together with the polynomial function.
    # Report the MSE and the theta vector.
    rawdata = []
    for pair in all_pairs:
        chosen_x = pair[0]
        chosen_y = pair[1]
        col_id_x = column_to_id[chosen_x]
        col_id_y = column_to_id[chosen_y]
        data = smartwatch_data[:, col_id_x]
        y = smartwatch_data[:, col_id_y]
        for i in range(2, 8):
            K = i
            design_matrix_x = compute_polynomial_design_matrix(x=data, K=K)
            
            theta_star = fit_multiple_lin_model(X=design_matrix_x, y=y)
            mse_loss = multiple_loss(X=design_matrix_x, y=y, theta=theta_star)
            x_label = chosen_x.replace("_", " ").capitalize()
            y_label = chosen_y.replace("_", " ").capitalize()
            plot_title = f"{x_label} vs. {y_label} / K={K}"
            figname = f"poly_{K}_{chosen_x}_vs_{chosen_y}"
            df = {"xlabel": x_label, "ylabel": y_label, "theta": theta_star, "mse": mse_loss, "K": K}
            rawdata.append(df)
            plot_scatterplot_and_polynomial(
                x=data,
                y=y,
                theta=theta_star,
                xlabel=x_label,
                ylabel=y_label,
                title=plot_title,
                figname=figname
            )
            #print(f"MSE: {mse_loss}; Theta: {theta_star}")
    df = pd.DataFrame(rawdata)
    df.to_excel("polynomial_regression.xlsx")


def task_2():
    print('\n---- Task 2 ----')

    for task in [1, 2, 3]:
        print(f'---- Logistic regression task {task} ----')
        if task == 1:
            # Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-1.npy')
            create_design_matrix = create_design_matrix_dataset_1
        elif task == 2:
            # Load the data set 2 (X-1-data.npy and targets-dataset-2.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-2.npy')
            create_design_matrix = create_design_matrix_dataset_2
        elif task == 3:
            # Load the data set 3 (X-2-data.npy and targets-dataset-3.npy)
            X_data = np.load('data/X-2-data.npy')
            y = np.load('data/targets-dataset-3.npy')
            create_design_matrix = create_design_matrix_dataset_3
        else:
            raise ValueError('Task not found.')

        X = create_design_matrix(X_data)

        # Plot the datapoints (just for visual inspection)
        plot_datapoints(X, y, f'Targets - Task {task}')

        # TODO: Split the dataset using the `train_test_split` function.
        # The parameter `random_state` should be set to 0.
        X_train, X_test, y_train, y_test = None, None, None, None

        print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Train the classifier
        custom_params = logistic_regression_params_sklearn()
        clf = LogisticRegression(**custom_params)
        # TODO: Fit the model to the data using the `fit` method of the classifier `clf`

        acc_train, acc_test = None, None # TODO: Use the `score` method of the classifier `clf` to calculate accuracy

        print(f'Train accuracy: {acc_train * 100:.2f}%. Test accuracy: {100 * acc_test:.2f}%.')
        
        yhat_train = None # TODO: Use the `predict_proba` method of the classifier `clf` to
                          #  calculate the predicted probabilities on the training set
        yhat_test = None # TODO: Use the `predict_proba` method of the classifier `clf` to
                         #  calculate the predicted probabilities on the test set

        # TODO: Use the `log_loss` function to calculate the cross-entropy loss
        #  (once on the training set, once on the test set).
        #  You need to pass (1) the true binary labels and (2) the probability of the *positive* class to `log_loss`.
        #  Since the output of `predict_proba` is of shape (n_samples, n_classes), you need to select the probabilities
        #  of the positive class by indexing the second column (index 1).
        loss_train, loss_test = None, None
        print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

        plot_logistic_regression(clf, create_design_matrix, X_train, f'(Dataset {task}) Train set predictions',
                                 figname=f'logreg_train{task}')
        plot_logistic_regression(clf, create_design_matrix, X_test,  f'(Dataset {task}) Test set predictions',
                                 figname=f'logreg_test{task}')

        # TODO: Print theta vector (and also the bias term). Hint: Check the attributes of the classifier
        classifier_weights, classifier_bias = None, None
        print(f'Parameters: {classifier_weights}, {classifier_bias}')


def task_3(initial_plot=True):
    print('\n---- Task 3 ----')
    # Do *not* change this seed
    np.random.seed(46)

    # TODO: Choose a random starting point using samples from a standard normal distribution
    x0 = None
    y0 = None
    print(f'Starting point: {x0:.4f}, {y0:.4f}')

    if initial_plot:
        # Plot the function to see how it looks like
        plot_3d_surface(rastrigin)
        plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0))   

    # TODO: Call the function `gradient_descent` with a chosen configuration of hyperparameters,
    #  i.e., learning_rate, lr_decay, and num_iters. Try out lr_decay=1 as well as values for lr_decay that are < 1.
    x_list, y_list, f_list = None, None, None

    # Print the point that is found after `num_iters` iterations
    print(f'Solution found: f({x_list[-1]:.4f}, {y_list[-1]:.4f})= {f_list[-1]:.4f}' )
    print(f'Global optimum: f(0, 0)= {rastrigin(0, 0):.4f}')

    # Here we plot the contour of the function with the path taken by the gradient descent algorithm
    plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0), 
                    x_list=x_list, y_list=y_list)

    # TODO: Create a plot f(x_t, y_t) over iterations t by calling `plot_function_over_iterations` with `f_list`
    pass


def main():
    np.random.seed(46)

    task_1(use_linalg_formulation=False)
    task_1(use_linalg_formulation=True)
    #task_2()
    #task_3(initial_plot=True)


if __name__ == '__main__':
    main()

