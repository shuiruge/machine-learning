# Exercise 2.10 of Machine Learning by T. Mitchell
# -----------------------------------------------
# Some problems are left:

def most_special_hypothesis():
    return ["empty", "empty", "empty", "empty", "empty", "empty"]
def hypothesis_apply(hypothesis, instance):
    """ list * list -> boolean
    """
    result = True
    i = 0
    while(result == True and i <= len(hypothesis) - 1):
        if hypothesis[i] == "empty":
            result = False
        elif hypothesis[i] == "?":
            result = True
        else:
            if hypothesis[i] == instance[i]:
                result = True
            else:
                result = False
        i += 1
    return result
def sample_instance(sample):
    return sample[0]
def sample_c(sample):
    return sample[1]
def generalize_hypothesis_by_sample(hypothesis_init, sample):
    """ [initial_hypothesis] -> [generalized_hypothesis]
    """
    def consistent(hypothesis, sample):
       if hypothesis_apply(hypothesis, instance) == sample_c(sample): 
           return True
       else:
           return False
    instance = sample_instance(sample)
    c = sample_c(sample)
    result = hypothesis_init[:]
    if consistent(hypothesis_init, sample) == True:
        return result 
    else:
        if c == True:
            for i in range(len(hypothesis_init)):
                if hypothesis_init[i] != instance[i]:
                    if hypothesis_init[i] == "empty":
                        result[i] = instance[i]
                    else:
                        result[i] = "?"
        return result

def find_s(samples):
    """ [samples] -> [hypothesis]
    
    ## test by section 2.4:
    >>> samples = [[["Sunny", "Warm", "Normal", "Strong", "Warm", "Same"], True],
                  [["Sunny", "Warm", "High", "Strong", "Warm", "Same"], True],
                  [["Rainy", "Cold", "High", "Strong", "Warm", "Change"], False],
                  [["Sunny", "Warm", "High", "Strong", "Cool", "Change"], True]]
        print(find_s(samples))
    >>> ['Sunny', 'Warm', '?', 'Strong', '?', '?']
    """
    s = most_special_hypothesis()
    for sample in samples:
        s = generalize_hypothesis_by_sample(s, sample)
    return s
## test by section 2.4:
#samples = [[["Sunny", "Warm", "Normal", "Strong", "Warm", "Same"], True],
#           [["Sunny", "Warm", "High", "Strong", "Warm", "Same"], True],
#           [["Rainy", "Cold", "High", "Strong", "Warm", "Change"], False],
#           [["Sunny", "Warm", "High", "Strong", "Cool", "Change"], True]]
#print(find_s(samples))

import random
def random_sample_generator(target_conception, num_of_samples):
    """ [target_conception] * Int -> samples = [[[instance], True],
                                                 [instance], True],
                                                 ...]
    remark: (i) for studying FIND-S, only positive samples are needed;
            (ii) all samples shall be different from each other.
    
    E.g. `random_sample_generator(["Sunny", "Warm", "?", "?", "?", "?"], 2)`
    may output `[[["Sunny", "Warm", "Normal", "Strong", "Cold", "Same"], True],
                 [["Sunny", "Warm", "High", "Strong", "Warm", "Same"], True]]`
    """
    conditions = [["Sky", ["Sunny", "Cloudy", "Rainy"]],
                  ["AirTemp", ["Warm", "Cold"]],
                  ["Humidity", ["Normal", "High"]],
                  ["Wind", ["Strong", "Weak"]],
                  ["Water", ["Warm", "Cold"]],
                  ["Forecase", ["Same", "Change"]]]
    if num_of_samples > 2 ** target_conception.count("?"):
        print("Error from random_sample_generator: num_of_samples exceeds its theoratical bound!")
    else:
        result = []
        j = 1
        while(j <= num_of_samples):
            instance = ["" for i in range(6)]
            for i in range(6):
                if target_conception[i] == "?":
                    instance[i] = random.choice(conditions[i][1])
                else:
                    instance[i] = target_conception[i]
            sample = [instance, True]
            if sample not in result:
                result.append(sample)
                j += 1
        return result
## test:
#print(random_sample_generator(["Sunny", "Warm", "?", "?", "?", "?"], 2))

def exercise_show(num_of_samples):
    """ Int --> Boolean
    """
    target_conception = ["Sunny", "Warm", "Normal", "?", "?", "?"]
    result = True
    i = 1
    while(i <= 20 and result == True):
        samples = random_sample_generator(target_conception, num_of_samples)
        s = find_s(samples)
        if s == target_conception:
            result = True
        else:
            result = False
        i += 1
    if result == True:
        print("s by FIND-S is always the target conception.")
    else:
        print("s by FIND-S is NOT always target conception.")

## Answer to exercise 2.10:
## By running `exercise_show(num_of_samples)` and tuning `num_of_samples` so that s by FIND-S = target conception, we find to the questions in this exercise:
## (i)   8 samples;
## (ii)  increase, e.g. if replace "Warm" -> "?", then the result becomes 10 samples; and if replace the 3rd "?" -> "Normal", then it becomes 5 samples.
## (iii) increase, since increasing features is increasing the number of "?" in target conception.
