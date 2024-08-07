import numpy as np

STABILITY_CONSTANT = 1e-8

class CostFunctions:
    def __init__(self):
        # dictionary (of tuples) with all activation functions and their derivatives
        self.__cost_functions = {
            'quadratic': CostFunctions.quadratic_cost_derivative,
            'huber': CostFunctions.huber_loss_derivative,
            'cross_entropy': CostFunctions.cross_entropy_derivative
        }

    @property
    #cost functions property
    def Cost_functions(self):
        return self.__cost_functions

    @staticmethod
    #Note: A more descriptive name might be get_cost_derivative_by_name
    def get_cost_function_by_name(func_name: str):
        cost_function_generator = CostFunctions()
        if func_name in cost_function_generator.Cost_functions:
            return cost_function_generator.Cost_functions[func_name]
        else:
            raise ValueError(f'Cost function {func_name} not recognized')

    @staticmethod
    # Quadratic cost function derivative
    def quadratic_cost_derivative(output_activations, y):
        y = y.reshape(-1, 1)
        return (output_activations - y)
    
    @staticmethod
    # Huber loss function derivative
    def huber_loss_derivative(output_activations, y):
        y = y.reshape(-1, 1)
        return np.where(np.abs(output_activations - y) <= 1, output_activations - y, np.sign(output_activations - y))
    
    @staticmethod
    def cross_entropy_derivative(output_activations, y):
        y = y.reshape(-1, 1)
        return (output_activations - y) / (output_activations * (1 - output_activations) + STABILITY_CONSTANT)