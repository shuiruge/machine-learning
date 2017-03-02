## This is after reading ?, ch2?.

## ========= Data Structure =========
## Instance = [Real]
## Training_datum = [Instance, Target_set]
## Training_data = [Training_datum]

## Selectors:
def instance(training_datum):
    return training_datum[0]
def target_concept(training_datum):
    return training_datum[1]

## ========= Adjust Raw Data =========

## Nomalilzation:
def features_max(training_data):
    """ Training_data -> [Real]
        training_data -> [max of feature for features of training_data]
    """
    dimension_of_instance_space = len(instance(training_data[0]))
    result = [-inf for i in range(dimension_of_instance_space)]
    for datum in training_data:
        for i in range(dimension_of_instance_space):
            if instance(datum)[i] > result[i]:
                result[i] = instance(datum)[i]
    return result

def features_min(training_data):
    """ Training_data -> [Real]
        training_data -> [min of feature for features of training_data]
    """
    dimension_of_instance_space = len(instance(training_data[0]))
    result = [inf for i in range(dimension_of_instance_space)]
    for datum in training_data:
        for i in range(dimension_of_instance_space):
            if instance(datum)[i] < result[i]:
                result[i] = instance(datum)[i]
    return result

## Store `features_boundaries` in RAM:
## Initialize `features_boundaries`:
features_boundaries = {'min': -1, 'max': 1}
def update_features_boundaries(raw_training_data):
    features_boundaries['min'] = features_min(raw_training_data)
    features_boundaries['max'] = features_max(raw_training_data)
    
def normalize_raw_instance(raw_instance):
    """ Instance -> Instance
    """
    dimension_of_instance_space = len(raw_instance)
    result = [0 for i in range(dimension_of_instance_space)]
    for i in range(dimension_of_instance_space):
        result[i] = (raw_instance[i] - features_boundaries['min'][i]) / (features_boundaries['max'][i] - features_boundaries['min'][i])
    return result

def normalize_raw_training_data(raw_training_data):
    """ Training_data -> Training_data
    """
    result = []
    for raw_datum in raw_training_data:
        result.append([normalize_raw_instance(instance(raw_datum)), target_concept(raw_datum)])
    return result


## ========= kNN =========
from math import sqrt, inf

def distance(point_1, point_2):
    dim = len(point_1)
    return sqrt(sum([(point_1[i] - point_2[i]) ** 2 for i in range(dim)]))

def sort_by_function(lst, function):
    """ [a] * (a -> Real) -> [b]
    where [a] is sorted to be [b] in order of function(a), descent by default.
    """
    lst_reconstruct = [(function(item), item) for item in lst]
    get_key = lambda item: item[0]
    sorted_lst = sorted(lst_reconstruct, key = get_key, reverse=True)
    result = [item[1] for item in sorted_lst]
    return result

def count(lst, fulfill_condition):
    """ [a] * (a -> Boolean) -> Int
    """
    satisfied = [_ for _ in lst if fulfill_condition(_) == True]
    return len(satisfied)

def kNN(k, training_data):
    """ Int * Training_data -> ([Real] -> Boolean)
        training_data -> (normalized_instance -> target_concept)
    """
    conceptions = set([target_concept(training_datum) for training_datum in training_data])
    def kNN_result(instance0):
        distance_to_instance = lambda training_datum: distance(instance(training_datum), instance0)
        sorted_training_data = sort_by_function(training_data, distance_to_instance) # descent by default.
        nearest_training_data = sorted_training_data[-k:]
        vote = lambda conception: count(nearest_training_data, lambda datum: target_concept(datum) == conception)
        conceptions_sorted_by_vote = sort_by_function(conceptions, vote)
        return conceptions_sorted_by_vote[0]
    return kNN_result