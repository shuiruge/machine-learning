## test:
scale = [2, 2, 1]
o_func = lambda x: x
ann = initialize_ann(scale, o_func)
w0 = ann['init_weights']

training_data = [{'input': [1, 0], 'target':[1]},
                 {'input': [0, 1], 'target':[0]}]
learning_rate = 0.3
times_of_training = 1
w = w0[:]

for sample in training_data:
    o = ann['outputs_from_units'](w, sample['input'])
    def generate_x():
        x = [None]
        x.append(sample['input'].append(1))
        for layer in range(1, len(scale) - 1):
            x.append(o[layer].append(1))
        return x
    x = generate_x()
#    def generate_delta():
    o.reverse()
    w_rev = w[:].reverse()
    delta = []
    t = sample['target']
    delta_output = [o[0][unit] * (1 - o[0][unit]) * (t[unit] - o[0][unit]) for unit in range(len(t))]
    delta.append(delta_output)
    delta_previous = delta_output[:]
    delta_hidden = []
    for layer in range(1, len(o) - 1):
        delta_hidden = [o[layer][unit] * (1 - o[layer][unit]) * sum([w_rev[layer - 1][k][unit] * delta_output[k] for k in range(len(delta_previous))]) for unit in range(len(o[layer]))]
        delta.append(delta_hidden)
        delta_previous = delta_hidden[:]
        delta.reverse() # we shall reverse back for employing eq.(4.16)
#        return delta
#    delta = generate_delta()
    def update_w(w):
        result = w[:]
        for layer in range(1, len(w)):
            for unit in range(len(w[0])):
                result[layer][unit] = [w[layer][unit][i] + \
                       learning_rate * delta[layer][unit] * x[layer][unit][i]
                       for i in len(w[layer][unit])]
        return result
    w = update_w(w)
 