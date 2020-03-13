# MIT 6.034 Lab 5: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')
from parse import *


################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""

    if id_tree.is_leaf():
        return id_tree.get_node_classification()
    else:
        child_node = id_tree.apply_classifier(point)
        return id_tree_classify_point(point, child_node)

#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""

    result = {}

    for element in data:
        classification = classifier.classify(element)
        if classification in result.keys():
            result[classification].append(element)
        else:
            result[classification] = [element]
    return result


#### Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""

    branches = split_on_classifier(data, target_classifier)

    disorder = 0
    for key in branches:
        term = len(branches[key])/len(data)
        classification = -1 * term * log2(term)
        disorder += classification
    return disorder


def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    branches = split_on_classifier(data, test_classifier)

    average_disorder = 0
    for key in branches:
        average_disorder += (len(branches[key]) / len(data)) * branch_disorder(branches[key], target_classifier)
    return average_disorder


## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab5.py:

# for classifier in tree_classifiers:
#     print(classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type")))


#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""

    score = 100000000
    winner = None

    for classifier in possible_classifiers:
        branches = split_on_classifier(data, target_classifier)
        new_score = average_test_disorder(data, classifier, target_classifier)
        if new_score < score:
            score = new_score
            winner = classifier
    if len(split_on_classifier(data, winner)) == 1:
        raise NoGoodClassifiersError
    return winner


## To find the best classifier from 2014 Q2, Part A, uncomment:
# print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""

    if id_tree_node == None:        
        id_tree_node = IdentificationTreeNode(target_classifier)

    if branch_disorder(data, target_classifier) == 0:
        id_tree_node.set_node_classification(target_classifier.classify(data[0]))
        return id_tree_node
    else:
        try:
            best_classifier = find_best_classifier(data, possible_classifiers, target_classifier)
            groups = split_on_classifier(data, best_classifier)
            new_id_tree_node = id_tree_node.set_classifier_and_expand(best_classifier, groups)
            branches = new_id_tree_node.get_branches()
            possible_classifiers.remove(best_classifier)
            for key in new_id_tree_node.get_branches():
                construct_greedy_id_tree(groups[key], possible_classifiers, target_classifier, branches[key])
            return new_id_tree_node
        except:
            return None


## To construct an ID tree for 2014 Q2, Part A:
# print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
# tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
# print(id_tree_classify_point(tree_test_point, tree_tree))

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
# print(construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification")))
# print(construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class")))
# tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_binary)
# print(tree)
# tree.print_with_data(heart_training_data)

test_patient = {\
    'Age': 20, #int
    'Sex': 'F', #M or F
    'Chest pain type': 'asymptomatic', #typical angina, atypical angina, non-anginal pain, or asymptomatic
    'Resting blood pressure': 100, #int
    'Cholesterol level': 120, #int
    'Is fasting blood sugar < 120 mg/dl': 'Yes', #Yes or No
    'Resting EKG type': 'normal', #normal, wave abnormality, or ventricular hypertrophy
    'Maximum heart rate': 150, #int
    'Does exercise cause chest pain?': 'No', #Yes or No
    'ST depression induced by exercise': 0, #int
    'Slope type': 'flat', #up, flat, or down
    '# of vessels colored': 0, #float or '?'
    'Thal type': 'unknown', #normal, fixed defect, reversible defect, or unknown
}

# tree.print_with_data([test_patient])
# print(id_tree_classify_point(test_patient, tree))

#### Part 1E: Multiple choice ##################################################


# print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))

ANSWER_1 = "bark_texture"
ANSWER_2 = "leaf_shape"
ANSWER_3 = "orange_foliage"

print(construct_greedy_id_tree(binary_data, binary_classifiers, feature_test("Classification")))

ANSWER_4 = [2,3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = 'nN'
ANSWER_9 = 'nN'


#### OPTIONAL: Construct an ID tree with medical data ##########################

## Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = False

if DO_OPTIONAL_SECTION:
    from parse import *
    medical_id_tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_discrete)


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""

    sum = 0
    for index, first in enumerate(u):
        sum += first*v[index]
    return sum


def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    return math.sqrt(dot_product(v,v))

def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    sum = 0

    for e1, e2 in zip(point1, point2):
        sum += (e1-e2)**2
    
    return math.sqrt(sum)


def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    sum = 0

    for e1, e2 in zip(point1, point2):
        sum += abs(e1-e2)
    
    return sum


def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    sum = 0

    for e1, e2 in zip(point1, point2):
        if e1 != e2:
            sum += 1
    
    return sum



def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""

    return 1 - (dot_product(point1,point2) / (norm(point1) * norm(point2)))


#### Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    sorted_list = sorted(data, key= lambda element: (distance_metric(point, element), sorter(element)))
    return sorted_list[:k]

def sorter(input):
    string_list = [str(i) for i in input.coords]
    return int("".join(string_list)) 
    

def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""


    closest_points = get_k_closest_points(point, data, k, distance_metric)

    dict = {}
    for point in closest_points:
        if point.classification in dict:
            dict[point.classification] += 1
        else:
            dict[point.classification] = 1
    
    return max(dict, key=dict.get)



## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""

    correct_classifications = 0

    for point in data:
        training_set = data.copy()
        training_set.remove(point)

        classification = knn_classify_point(point, training_set, k, distance_metric)

        if classification == point.classification:
            correct_classifications += 1
    return correct_classifications / len(data)






def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""

    class TestSet:
        def __init__(self, k, distance, metric):
            self.k = k
            self.distance = distance
            self.metric = metric

    metrics = [euclidean_distance, manhattan_distance, hamming_distance, cosine_distance]

    results = []
    
    for metric in metrics:
        max_k = int((len(data)/2)+1)
        for k in range(1, max_k):
            results.append(TestSet(k, cross_validate(data, k, metric), metric))
    
    winner = max(results, key=lambda element: element.distance)
    return winner.k, winner.metric



## To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))


#### Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = None
kNN_ANSWER_2 = None
kNN_ANSWER_3 = None

kNN_ANSWER_4 = None
kNN_ANSWER_5 = None
kNN_ANSWER_6 = None
kNN_ANSWER_7 = None


#### SURVEY ####################################################################

NAME = "Maximilian Deichmann"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
