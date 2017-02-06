# Exercises 3.X of Machine Learning by T. Mitchell
# -----------------------------------------------
# scale e.g. [3, 2, 1, 3] represents the Ann with 3 input units, 2 hidden units, then 1 hidden unit, and 3 output units.
# initialized weights are random numbers in [-0.05, 0.05]
# Ann = {"weights": Weights, "output": output}
# Weights = [weights_on_layer]
# weight_on_layer = [weights_on_unit]
# weights_on_unit = [w_{up_stream_unit}, ..., threshold]
#   wherein the last is the threshold w_0
# output(weights, input) -> output: Weights * [input_i] -> [output_i]
# -----------------------------------------------

from copy import copy
from random import uniform 

def generate_initialized_weights(scale):
    """ -> [[[Real]]]
        
    weights[layer][up_stream_unit][unit]
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
        init_weights_on_layer = generate_table(random_w, scale[layer], scale[up_stream_layer] + 1) # caution the threshold!
        init_weights.append(init_weights_on_layer)
    return init_weights

def outputs_from_units(scale, o_func, weights, inpt):
    """ [Int] * (Real -> Real) * [[[Real]]] * [Real] -> [[Real]]
        
    outputs outputs of units on each layer
    """
    def output_from_unit(weights_on_unit, inpt):
        """ [Real] * [Real] -> Real
        """
        threshold = weights_on_unit[-1]
        net = sum([inpt[i] * weights_on_unit[i] for i in range(len(inpt))]) + \
              threshold
        return o_func(net)
    def outputs_from_layer(weights_on_layer, inpt):
        """ [[Real]] * [Real] -> [Real]
        """
        result = []
        for weights_on_unit in weights_on_layer:
            result.append(output_from_unit(weights_on_unit, inpt))
        return result
    result = [None] # layer = 0 does not output.
    x = inpt.copy()
    for layer in range(1, len(scale)):
        weights_on_layer = weights[layer]
        x = outputs_from_layer(weights_on_layer, x).copy()
        result.append(x)
    return result

def initialize_ann(scale, o_func):
    """ [Int] * (Real -> Real) -> Ann
    """
    return {"init_weights": generate_initialized_weights(scale),
            "outputs_from_units": lambda weights, inpt: outputs_from_units(scale, o_func, weights, inpt)}

def back_propagation(learning_rate, scale, o_func, training_data):
    """ [Int] * (Real -> Real) * [{'input': [Real], 'target': [Real]}]
    
    follow the notations in table 4-2
    """
    ann = initialize_ann(scale, o_func)
    w = ann['init_weights']
    def training():
        for sample in training_data:
            # 1
            o = ann['outputs_from_units'](w, sample['input'])
            def compute_x():
                x = [o[layer] for layer in range(1, len(scale) - 1)]
                x.prepend(sample['input'])
                x.prepend(None)
                return x
            x = comput_x()
            # it benefits to reverse since it's **back**-propagating
            o.reverse()
            def generate_delta():
                delta = []
                # 2
                t = sample['target']
                delta_output = [o[0][i] * (1 - o[0][i]) * (t[i] - o[0][i])
                                for i in range(len(t))]
                delta.append(delta_output)
                # 3
                delta_previous = delta_output[:]
                for h in range(1, len(o) - 1): # the last layer (the groud layer before reverse) is None.
                    delta_hidden = [o[h][i] * (1 - o[h][i]) * sum([w[k][h] * delta_output[k]
                                                                   for k in range(len(delta_previous))])
                                    for i in range(len(o[h]))]
                    delta.append(delta_hidden)
                    delta_previous = delta_hidden[:]
                delta.reverse() # we shall reverse back for employing eq.(4.16)
                return delta
            delta = generate_delta()
            # 4
            def update_w(w):
                result = w[:]
                for layer in range(1, len(w)):
                    for unit in range(len(w[0])):
                        result[layer][unit] = [w[layer][unit][i] + \
                                          learning_rate * delta[layer][unit] * x[layer][unit][i]
                                          for i in len(w[layer][unit])]
                return result
            w = update_w(w)
    max_step = 10
    for step in max_step:
        training()
    return w
## needs test!       
