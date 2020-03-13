# MIT 6.034 Lab 8: Bayesian Inference
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    ancestors = net.get_parents(var)
    for ancestor in ancestors:
        second_level_ancestors = get_ancestors(net, ancestor)
        ancestors = ancestors.union(second_level_ancestors)
    return ancestors

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    descendants = net.get_children(var)
    for descendant in descendants:
        second_level_descendants = get_descendants(net, descendant)
        descendants = descendants.union(second_level_descendants)
    return descendants

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    descendants = get_descendants(net, var)
    ancestors = get_ancestors(net, var)
    all_nodes = net.get_variables()
    a = set(all_nodes) - descendants
    a.remove(var)
    return ancestors.union(a)

#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    new_givens = givens.copy()
    if all(elem in new_givens for elem in net.get_parents(var)) and any(elem in get_descendants(net, var) for elem in new_givens) == False:
        for key in givens:
            if key in get_nondescendants(net, var) and key not in net.get_parents(var):
                del new_givens[key]
    return new_givens

def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    try:
        hypothesis_var = list(hypothesis.keys())[0]
        if givens is not None:
            givens = simplify_givens(net, hypothesis_var, givens)
        return net.get_probability(hypothesis, givens)
    except:
        raise LookupError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    joint_probability = 1

    for node in net.topological_sort():
        givens = {}
        parents = net.get_parents(node)
        for parent in parents:
            givens[parent] = hypothesis[parent]

        probability = probability_lookup(net, {node: hypothesis[node]}, givens)
        joint_probability *= probability
    return joint_probability

    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    return sum([probability_joint(net, combination) for combination in net.combinations(net.get_variables(), hypothesis)])

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    if givens is None:
        return probability_marginal(net, hypothesis)
    else:

        if any(givens[a] != hypothesis[b] for a in givens for b in hypothesis if a == b):
            return 0

        numerator = probability_marginal(net, dict(hypothesis, **givens))
        denumerator = probability_marginal(net, givens)

        return numerator / denumerator
    
def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    if all(required_variable in hypothesis for required_variable in net.get_variables()) and givens is None:
        return probability_joint(net, hypothesis)
    elif givens is None:
         return probability_marginal(net, hypothesis)
    else:
        return probability_conditional(net, hypothesis, givens)


#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """

    result = []
    variables = net.get_variables()

    for variable in variables: 
        value = 1 
        num_domains = len(net.get_domain(variable))-1
        for parent in net.get_parents(variable):
            num_parent_domains = len(net.get_domain(parent))
            value *= num_parent_domains
        result.append(num_domains * value)
    return sum(result)


#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """


    return all(approx_equal(probability(net, {var1: domain_var1}, givens) * probability(net, {var2: domain_var2}, givens), 
                            probability(net, {var1: domain_var1, var2: domain_var2}, givens)) 
                                for domain_var1 in net.get_domain(var1) for domain_var2 in net.get_domain(var2))

    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    raise NotImplementedError


#### SURVEY ####################################################################

NAME = "Maximilian Deichmann"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 5
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
