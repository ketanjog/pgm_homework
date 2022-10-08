class Logger:
    def __init__(self, verbose):
        self.verbose = True
        pass

    def print(self, text: str):
        if self.verbose:
            print(text)
        else:
            pass

    def print_progress(self, iteration, frequency=1):
        if self.verbose:
            # Print out progress every frequency number of iterations
            if iteration % frequency == 0:
                print("---------------------------------")
                print(f"The Log Likelihood is {self.log_likelihood_values[-1]:.2f} ")
                percent_increase = float(
                    (self.log_likelihood_values[-4] - self.log_likelihood_values[-1])
                    / self.log_likelihood_values[-1]
                )
                print(f"Likelihood increased by {percent_increase:.2f} percent")
                print("\n\n")
                print("----------------------------------")
