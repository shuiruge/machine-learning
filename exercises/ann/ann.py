# Exercises 3.X of Machine Learning by T. Mitchell
# -----------------------------------------------
# scale e.g. [3, 2, 1, 3] represents the Ann with 3 input units, 2 hidden units, then 1 hidden unit, and 3 output units.
# initialized weights are random numbers in [-0.05, 0.05]
# Ann = {"weights": Weights, "output": output}
# Weights = [nets_between_layers]
# nets_between_layers = [w_{init_unit, end_unit}, ...]
#   wherein the last init_unit of w is for the thresheld w_0
# output(weights, input) -> output: Weights * [input_i] -> [output_i]
# -----------------------------------------------

from copy import copy
from random import uniform 

def initialize_ann(scale, o_func):
    """ [Int] -> Ann
    """
    def generate_initialized_weights():
        """ -> [[[Real]]]
        
        weight[layer][up_stream_unit][unit]
          wherein, layer = 1,2,... since layer = 0 is for input-units.
        so we let weight[0] = None
        """
        def generate_table(func, rows, columns):
            """ (Int * Int -> Real) * Int * Int -> [[Real]]
            """
            result = []
            for i in range(rows):
                column = [func(i,j) for j in range(columns)]
                result.append(column)
            return result
        def random_w(i, j):
            """ Int * Int -> Real
            """
            return uniform(-0.05, 0.05)
        init_weights = [None]
        for layer in range(1, len(scale)):
            up_stream_layer = layer - 1
            init_weights_on_layer = generate_table(random_w, scale[up_stream_layer] + 1, scale[layer]) # caution the w_0!
            init_weights.append(init_weights_on_layer)
        return init_weights
    def outputs_from_units(weights, inpt):
        """ [[[Real]]] * [Real] -> [[Real]]
        
        outputs outputs of units on each layer
        """
        def output_from_unit(weights_on_unit, inpt):
            """ [Real] * [Real] -> Real
            """
            net = sum([inpt[i] * weights_on_unit[i] for i in range(len(inpt))]) + \
                  weights_on_unit[len(inpt)]
            return o_func(net)
        def outputs_from_layer(layer, weights_on_layer, inpt):
            """ Int * [[Real]] * [Real] -> [Real]
            
            the first layer shall be layer = 1, since the
            groud layer, i.e. layer = 0 does not output.
            """
            result = []
            for unit in range(scale[layer]):
                weights_on_unit = [weights_on_layer[up_stream_unit][unit]
                                   for up_stream_unit in range(len(inpt) + 1)]
                result.append(output_from_unit(weights_on_unit, inpt))
            return result
        result = [None] # layer = 0 does not output.
        x = inpt.copy()
        for layer in range(1, len(scale)):
            weights_on_layer = weights[layer]
            x = outputs_from_layer(layer, weights_on_layer, x).copy()
            result.append(x)
        return result
    return {"init_weights": generate_initialized_weights(), "outputs_from_units": outputs_from_units}

