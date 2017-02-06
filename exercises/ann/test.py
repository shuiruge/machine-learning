## test:
scale = [2, 2, 1]
o_func = lambda x: x
ann = initialize_ann(scale, o_func)
w = ann['init_weights']
a = ann['outputs_from_units']
print(w)
print()

print(a(w, [1, 2]))
