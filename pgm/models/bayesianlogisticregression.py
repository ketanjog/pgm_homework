"""
Class for Bayesian Logistic Regression. 

Trying to follow David Blei's Probablistic Graphical Models Class notes
as closely as possible. 
"""

import numpy as np
from pgm.utils.math import sigmoid
from pgm.utils.logger import Logger
from pgm.data.load_imdb import load_processed_imdb_data
import matplotlib.pyplot as plt


class BayesianLogisticRegression:
    def __init__(self, prior_variance: float, learning_rate: float = 0.01):
        """
        Initialisation parameters:

        prior_variance : float = the variance on the Bernoulli prior for the betas
        learning_rate : float = the step size for stochastic gradient ascent

        """
        self.prior_variance = prior_variance
        self.learning_rate = learning_rate
        self.beta: np.ndarray = None

        self.features: np.ndarray = None
        self.responses: np.ndarray = None
        self.term_1 = None

        # Storing data for plotting
        self.log_likelihood_values = []
        self.accuracy_values = []
        self.grad_values = []

    def fit(self, features: np.ndarray, responses: np.ndarray, verbose=True):
        """
        Fits the Bayesian Logistic model to the dataset
        """
        # Load the features and responses into class variables
        split = int(len(responses) * 0.8)

        # Shuffle and choose random subset of data
        new_index = np.random.permutation(len(responses))

        # Training set
        self.train_features = features[new_index][:split]
        self.train_responses = responses[new_index][:split]

        # Testing set
        self.test_features = features[new_index][split:]
        self.test_responses = responses[new_index][split:]

        self.beta = np.zeros([features.shape[1]])
        lg = Logger(verbose=verbose)

        # assert self.features.shape == (50001, 100)

        # run stochastic gradient ascent on the log joint
        lg.print("Running stochastic gradient ascent")
        self.stochastic_gradient_ascent(verbose=verbose)
        lg.print("Fit Successful!")

        # return self.beta

    def predict(self, data_train: np.ndarray):
        """
        Returns E[response | new_features, beta_fitted]
        """
        if self.train_features is None or self.train_responses is None:
            raise ValueError("Fit the model on data first")
        response = sigmoid(np.matmul(data_train, self.beta))
        response[response > 0.5] = 1
        response[response <= 0.5] = 0

        return np.asarray(response).reshape(-1)

    def objective(self):
        """
        Return log of the joint
        """
        if self.features is None or self.responses is None:
            raise ValueError("Fit the model on data first")

        term_1 = np.matmul(self.features, self.beta)
        self.term_1 = term_1

        return (
            np.dot(np.log(sigmoid((term_1))).T, self.responses)
            + np.dot(np.log(sigmoid((-term_1))).T, (1 - self.responses))
            - self.prior_variance * np.sum(self.beta**2)
        )
        # test with one objective

    def grad_objective(self, batch: int):
        """
        returns the gradient of the objective using the current beta
        given the features and response data
        """
        if self.features is None or self.responses is None:
            raise ValueError("Fit the model on data first")

        deviation = (
            np.dot(self.features.T, (self.responses - sigmoid(self.term_1))) / batch
        )
        regulariser = self.prior_variance * self.beta
        gradient = -regulariser + deviation

        # assert gradient.shape[0] == self.beta.shape[0]

        return gradient

    def evaluate(self):
        self.y_hat = self.predict(self.test_features)
        self.test_size = len(self.test_responses)

        y_hat = self.predict(self.test_features)
        test_size = len(self.test_responses)

        # print(str(np.sum(np.absolute(y_hat - self.test_responses))))
        accuracy = (
            test_size - np.sum(np.absolute(y_hat - self.test_responses))
        ) / test_size

        return accuracy

    def stochastic_gradient_ascent(self, batch=100, smart_stopping=True, verbose=True):
        """
        Performs stochastic gradient ascent on the data
        to fit beta
        """
        lg = Logger(verbose=verbose)
        if self.train_features is None or self.train_responses is None:
            raise ValueError("Fit the model on data first")

        # Initialise beta to be zeros
        self.beta = np.zeros_like(self.beta)
        self.beta = self.beta.reshape((len(self.beta), 1))

        # Iterating until convergence
        if smart_stopping:

            iterations = 0
            MAX_ITERATIONS = 10000

            while True:

                # Shuffle and choose random subset of data
                new_index = np.random.permutation(batch)
                self.features = self.train_features[new_index]
                self.responses = self.train_responses[new_index].reshape(-1, 1)

                # Save old likelihood
                new_likelihood = self.objective()

                old_likelihood = (
                    0
                    if len(self.log_likelihood_values) == 0
                    else self.log_likelihood_values[-1]
                )

                # Check for convergence
                if abs(new_likelihood - old_likelihood) < 0.0001:
                    lg.print_progress(iterations, frequency=1)
                    break

                # Save new likelihood value
                self.log_likelihood_values.append(new_likelihood)
                self.accuracy_values.append(self.evaluate())
                self.grad_values.append(self.grad_objective(batch))

                # Update beta
                grad = self.grad_objective(batch=batch)

                self.beta += np.multiply(self.learning_rate, grad)
                iterations += 1

                # Print progress nicely
                # lg.print_progress(iterations)
                if iterations % 10 == 0:
                    print(
                        f" Log Likelihood is{self.log_likelihood_values[-1].item():.2f} "
                    )
                    print(f" Accuracy is{self.accuracy_values[-1]:.2f} ")

                # For testing, if stop if we exceed max iterations
                if iterations > MAX_ITERATIONS:
                    print("Reached maximum iterations. Breaking out of optimisation")
                    break

    def plot_likelihood(self):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.title("Likelihood")
        plt.plot(np.asarray(self.log_likelihood_values).reshape(-1), color="red")

        plt.show()

    def plot_accuracy(self):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.title("Accuracy")
        plt.plot(self.accuracy_values, color="red")

        plt.show()
