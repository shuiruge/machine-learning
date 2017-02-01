# Exercise 3.4 of Machine Learning by T. Mitchell
# -----------------------------------------------
# Sample is a Dict (table), as in the texture:
#     Sample = {attribute_1: value, ..., attribute_n: value, target_conception: True or False}
# Leave = {"class": "Leave", "leave": attribute_value, "samples": [Sample], "value": devoted_value}
# Vertex = {"class": "Vertex", "vertex": attribute, "leaves": [Leave]}
# Tree = root_Vertex
#   where, in `root_Vertex`, `Leave` maybe replaced by `Vertex`, and this replacement is recursive.
# -----------------------------------------------
# Problem:
#    It seems that Python is NOT applicative order!!!
#    Indeed, if drop `copy_list(samples)` and replace `samples0` -> `samples` in `drop_attribute`,
#    then the input will be affected after running `drop_attribute`, which shall NOT happen for
#    applicative order.




## Sample:
def copy_samples(samples):
    """ [Sample] -> [Sample]
    """
    return [sample.copy() for sample in samples]

def sample_attributes(sample):
    """ Sample -> [Str]
    """
    raw_attributes = [key for key in sample]
    raw_attributes.remove("target_conception")
    return raw_attributes

def attribute_values(samples, attribute):
    """ [Sample] * Str -> [a]
    """
    result = []
    for sample in samples:
        attribute_value = sample[attribute]
        if attribute_value in result:
            result = result
        else:
            result.append(attribute_value)
    return result

def seperate_samples_by_attribute(samples, attribute):
    """ [Sample] * Str -> (a -> [Sample])
    """
    def S_v(attr_value):
        result = []
        for sample in samples:
            if sample[attribute] == attr_value:
                result.append(sample)
            else:
                result = result
        return result
    return S_v

## entropy & gain:
from math import log
def entropy(samples):
    """ [Sample] -> Real
    """
    if len(samples) == 0:
        return 0
    else:
        def probability(elem, lst):
            return lst.count(elem) / len(lst)
        target_conceptions = [sample["target_conception"] for sample in samples]
        p_true = probability(True, target_conceptions)
        p_false= probability(False, target_conceptions)
        if p_true == 0 or p_false == 0:
            return 0
        else:
            return - p_true * log(p_true, 2) - p_false * log(p_false, 2)

def gain(samples, attribute):
    """ [Sample] * Str -> Real
    """
    attr_values = attribute_values(samples, attribute)
    S_v = seperate_samples_by_attribute(samples, attribute)
    return entropy(samples) - sum([((len(S_v(attr_value)) / len(samples)) * entropy(S_v(attr_value)))
                                   for attr_value in attr_values])

## generate Vertex:
def drop_attribute(samples, attribute):
    """ [Sample] -> [Sample]
    
    It seems that Python is NOT applicative order!!!
    Indeed, if drop `copy_list(samples)` and replace
    `samples0` -> `samples`, then the input will be affected
    after running `drop_attribute`, which shall NOT happen
    for applicative order.
    """
    samples0 = copy_samples(samples)
    for sample in samples0:
        sample.pop(attribute)
    return samples0

def most_devoted(samples):
    """ [Sample] -> Boolean
    """
    num_true = 0
    num_false = 0
    for sample in samples:
        if sample["target_conception"] == True:
            num_true += 1
        else:
            num_false += 1
    if num_true >= num_false:
        return True
    else:
        return False

def generate_vertex(samples, attribute):
    """ [Sample] * Str -> Vertex
    
    In the output, samples in Leave will not contain the `attribute`,
    as demanded by ID3.
    """
    attr_values = attribute_values(samples, attribute)
    S_v = seperate_samples_by_attribute(samples, attribute)
    leaves = []
    for attr_value in attr_values:
        leave = {"class": "Leave",
                 "leave": attr_value,
                 "samples": drop_attribute(S_v(attr_value), attribute),
                 "value": most_devoted(S_v(attr_value))}
        leaves.append(leave)
    return {"class": "Vertex",
            "vertex": attribute,
            "leaves": leaves}

def to_be_vertexQ(leave):
    """ Leave -> Boolean
    """
    samples = leave["samples"]
    if entropy(samples) == 0:
        return False
    else:
        attributes = sample_attributes(samples[0])
        if len(attributes) == 1:
            return False
        else:
            return True 
        
def max_gain_attribute(samples):
    """ [Sample] -> Str
    """
    attributes = sample_attributes(samples[0])
    result_attribute = attributes[0]
    gain_max = gain(samples, result_attribute)
    for attribute in attributes:
        gain_new = gain(samples, attribute)
        if gain_new > gain_max:
            gain_max = gain_new
            result_attribute = attribute
        else:
            result_attribute = result_attribute
    return result_attribute
  
def deal_with_leave(leave):
    """ Leave -> Leave, or Vertex
    """
    if not to_be_vertexQ(leave):
        return leave
    else:
        samples = leave["samples"]
        new_vertex_attribute = max_gain_attribute(samples)
        return generate_vertex(samples, new_vertex_attribute)

def update_leaves(vertex):
    """ Vertex (or Leave) -> Vertex (or Leave)
    """
    if vertex["class"] == "Leave":
        return deal_with_leave(vertex)
    else:
        sub_vertices = vertex["leaves"]
        return {"class": "Vertex",
                "vertex": vertex["vertex"],
                "leaves": [update_leaves(sub_vertex) for sub_vertex in sub_vertices]}

def fixed_point(func, init_x):
    """ (a -> a) * a -> a
    """
    if func(init_x) == init_x:
        return init_x
    else:
        return fixed_point(func, func(init_x))

def id3(samples):
    """ [Sample] -> Vertex
    """
    def generate_root_vertex(samples):
        """ [Sample] -> Vertex
        """
        root_attribute = max_gain_attribute(samples)
        root_attr_values = attribute_values(samples, root_attribute)
        S_v = seperate_samples_by_attribute(samples, root_attribute)
        leaves = []
        for attr_value in root_attr_values:
            leave = {"class": "Leave",
                     "leave": attr_value,
                     "samples": drop_attribute(S_v(attr_value), root_attribute),
                     "value": most_devoted(S_v(attr_value))}
            leaves.append(leave)
        return {"class": "Vertex",
                "vertex": root_attribute,
                "leaves": leaves}
    root_vertex = generate_root_vertex(samples)
    id3_generated_tree = fixed_point(update_leaves, root_vertex)
    return id3_generated_tree

## Test:
## Table 3-2:
D1 = {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "target_conception": False}
D2 = {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "target_conception": False}
D3 = {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "target_conception": True}
D4 = {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "target_conception": True}
D5 = {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "target_conception": True}
D6 = {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "target_conception": False}
D7 = {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "target_conception": True}
D8 = {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "target_conception": False}
D9 = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "target_conception": True}
D10 = {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "target_conception": True}
D11 = {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "target_conception": True}
D12 = {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "target_conception": True}
D13 = {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "target_conception": True}
D14 = {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "target_conception": False}
samples = [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14]

id3_tree = id3(samples)
## which is consistent with Figure 3-1.

## Finally, exercise 3.4:
## Table 2-1:
D1 = {"Sky": "Sunny", "AirTemp": "Warm", "Humidity": "Normal", "Wind": "Strong", "Water": "Warm", "Forecast": "Same", "target_conception": True}
D2 = {"Sky": "Sunny", "AirTemp": "Warm", "Humidity": "High", "Wind": "Strong", "Water": "Warm", "Forecast": "Same", "target_conception": True}
D3 = {"Sky": "Rainy", "AirTemp": "Cold", "Humidity": "Normal", "Wind": "Strong", "Water": "Warm", "Forecast": "Change", "target_conception": False}
D4 = {"Sky": "Sunny", "AirTemp": "Warm", "Humidity": "High", "Wind": "Strong", "Water": "Cool", "Forecast": "Change", "target_conception": True}
samples = [D1, D2, D3, D4]

id3_tree = id3(samples)
