# MIT 6.034 Lab 7: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce
import math


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    sum = 0
    for index, first in enumerate(u):
        sum += first*v[index]
    return sum

def norm(v):
    """Computes the norm (length) of a vector v, represented 
    as a tuple or list of coords."""
    return math.sqrt(dot_product(v,v))


#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    return dot_product(svm.w, point.coords) + svm.b

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    leng = positiveness(svm, point)
    if leng == 0:
        return 0
    elif leng > 0:
        return 1
    else:
        return -1

def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    return 2/norm(svm.w)
    

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""

    violating_points = []

    for training_point in svm.training_points:
        if training_point in svm.support_vectors:
            if training_point.classification != positiveness(svm, training_point):
                violating_points.append(training_point)
        
        if positiveness(svm, training_point) < 1 and positiveness(svm, training_point) > -1:
            violating_points.append(training_point)

    return set(violating_points)

#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""

    violating_points = []

    for point in svm.training_points:
        
        if point in svm.support_vectors and point.alpha <= 0:
            violating_points.append(point)
        
        if point not in svm.support_vectors and point.alpha != 0:
            violating_points.append(point)
    
    return set(violating_points)

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""

    four_sum = 0
    w_calculated = []

    for point in svm.training_points:
        four_sum += classify(svm, point) * point.alpha
        w_calculated.append(scalar_mult(point.classification * point.alpha, point.coords))

    w_calculated = [sum(i) for i in zip(*w_calculated)]

    return four_sum == 0 and w_calculated == svm.w

#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    return set([point for point in svm.training_points if classify(svm, point) != point.classification])

#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""

    svm.support_vectors = [point for point in svm.training_points if point.alpha > 0]
    
    w_calculated = []
    for point in svm.training_points:
        w_calculated.append(scalar_mult(point.classification * point.alpha, point.coords))
    svm.w = [sum(i) for i in zip(*w_calculated)]

    b_min = 100000
    b_max = -100000

    for point in svm.support_vectors:
        b = - dot_product(svm.w, point) + point.classification
        if point.classification == 1 and b > b_max:
            b_max = b
        if point.classification == -1 and b < b_min:
            b_min = b
    svm.b = (b_min + b_max) / 2

    return svm


#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = 'AD'
ANSWER_6 = 'ABD'
ANSWER_7 = 'ABD'
ANSWER_8 = []
ANSWER_9 = 'ABD'
ANSWER_10 = 'ABD'

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1,3,6,8]
ANSWER_18 = [1,2,4,5,6,7,8]
ANSWER_19 = [1,2,4,5,6,7,8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = "Maximilian Deichmann"
COLLABORATORS = "nonw"
HOW_MANY_HOURS_THIS_LAB_TOOK = 5
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
