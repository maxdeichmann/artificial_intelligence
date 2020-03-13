# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    return {point: make_fraction(1,len(training_points)) for point in training_points}

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""

    dictionary = {}

    for point in classifier_to_misclassified:
        score = make_fraction(0)
        for misclassified_point in classifier_to_misclassified[point]:
            score = score + point_to_weight[misclassified_point]
        dictionary[point] = score
    
    return dictionary


def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    
    best_keys = []
    min_value = min(classifier_to_error_rate.values()) 
    min_keys = [key for key in classifier_to_error_rate if classifier_to_error_rate[key] == min_value]

    if use_smallest_error:
        best_keys = min_keys
    else:
        max_value = max(classifier_to_error_rate.values())
        max_keys = [key for key in classifier_to_error_rate if classifier_to_error_rate[key] == max_value]

        dist_to_max = abs(max_value - make_fraction(1,2))
        dist_to_min = abs(min_value - make_fraction(1,2))
        
        if dist_to_max == dist_to_min:
            best_keys = min_keys + max_keys
        else:
            best_keys = max_keys if dist_to_max > dist_to_min else min_keys
    
    best_keys = sorted(best_keys)
    if classifier_to_error_rate[best_keys[0]] == make_fraction(1,2):
        raise NoGoodClassifiersError
    else:
        return best_keys[0]


def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""

    if error_rate == 0:
        return INF
    elif error_rate == 1:
        return -INF
    else:
        return 1/2 * ln(make_fraction(1-error_rate,error_rate))

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    missclassified_points = { i : 0 for i in training_points }
    for classifier in H:
        for training_point in training_points:
            if training_point in classifier_to_misclassified[classifier[0]]:
                missclassified_points[training_point] -= classifier[1]
            else:
                missclassified_points[training_point] += classifier[1]

    result = [k for k,v in missclassified_points.items() if v <= 0]
    return set(result)


def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""

    misclassified_points = get_overall_misclassifications(H, training_points, classifier_to_misclassified)
    return False if len(misclassified_points) > mistake_tolerance else True


def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    for key in point_to_weight:

        if key not in misclassified_points:            
            point_to_weight[key] = make_fraction(1,2) * make_fraction(1,(1-error_rate)) * make_fraction(point_to_weight[key])
        else:
            point_to_weight[key] = make_fraction(1/2) * make_fraction(1,error_rate) * make_fraction(point_to_weight[key])
    return point_to_weight

#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    H = []
    point_to_weights = initialize_weights(training_points)
    while max_rounds > 0:
        max_rounds -= 1
        try:
            classifier_to_error_rate = calculate_error_rates(point_to_weights, classifier_to_misclassified)
            current_classifier_name = pick_best_classifier(classifier_to_error_rate, use_smallest_error)
            voting_power = calculate_voting_power(make_fraction(classifier_to_error_rate[current_classifier_name]))
            current_classifier = (current_classifier_name, voting_power)
            H.append(current_classifier)
            if max_rounds is 0 or is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
                return H
            else:
                misclassified_points = classifier_to_misclassified[current_classifier_name]
                point_to_weights = update_weights(point_to_weights, misclassified_points, classifier_to_error_rate[H[-1][0]])
        except NoGoodClassifiersError as error:
            return H

#### SURVEY ####################################################################

NAME = "Maximilian Deichmann"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 4
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
