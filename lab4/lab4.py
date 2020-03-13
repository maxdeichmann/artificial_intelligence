# MIT 6.034 Lab 4: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    return True in [True if len(csp.domains[key]) == 0 else False for key in csp.domains]

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""

    for key1 in csp.assignments:
        for key2 in csp.assignments:
            constraints = csp.constraints_between(key1, key2)
            for constraint in constraints:
                if constraint.check(csp.assignments[key1], csp.assignments[key2]) == False:
                    return False
    return True



#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem) :
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    if check_all_constraints(problem) == False or has_empty_domains(problem) == True:
        return (None, 1)
    elif check_all_constraints(problem) == True and len(problem.unassigned_vars) == 0:
        return (problem.assignments, 1)
    else:
        winner = (None, 0)
        unassigned_var = problem.pop_next_unassigned_var()
        
        domains = problem.get_domain(unassigned_var)
        count = 1
        for domain in domains:
            new_problem = problem.copy()
            new_game = new_problem.set_assignment(unassigned_var, domain)
            solution = solve_constraint_dfs(new_game)
            count = count + solution[1]
            if solution[0] != None:
                winner = solution
                break
        winner = winner[0], count
        return winner



# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?
# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

ANSWER_1 = solve_constraint_dfs(get_pokemon_problem())[1]


#### Part 3: Forward Checking ##################################################

def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    v_domain = csp.get_domain(var)
    neighbors = csp.get_neighbors(var)

    reduced = []
    for neighbor in neighbors:

        n_domain = csp.get_domain(neighbor)

        removables = []

        for n_value in n_domain:
            constraints = csp.constraints_between(var, neighbor)

            evaluations = [True] * len(v_domain)
            for index, v_value in enumerate(v_domain):
                for constraint in constraints:
                    if constraint.check(n_value, v_value) == False:
                        evaluations[index] = False
            
            if not any(evaluations):
                reduced.append(neighbor)
                removables.append(n_value)


        for removable in removables:
            csp.eliminate(neighbor, removable)
            if len(csp.get_domain(var)) == 0 or len(csp.get_domain(neighbor)) == 0:
                return None

    return sorted(set(reduced))


# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    if check_all_constraints(problem) == False or has_empty_domains(problem) == True:
        return (None, 1)
    elif check_all_constraints(problem) == True and len(problem.unassigned_vars) == 0:
        return (problem.assignments, 1)
    else:
        winner = (None, 0)
        unassigned_var = problem.pop_next_unassigned_var()
        domains = problem.get_domain(unassigned_var)

        count = 1
        for domain in domains:
            new_problem = problem.copy()
            new_game = new_problem.set_assignment(unassigned_var, domain)
            eliminate_from_neighbors(new_game, unassigned_var)
            solution = solve_constraint_forward_checking(new_game)
            count = count + solution[1]
            if solution[0] != None:
                winner = solution
                break
        winner = winner[0], count
        return winner

# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?

ANSWER_2 = solve_constraint_forward_checking(get_pokemon_problem())[1]


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    deque = [] 

    if queue == None:
         queue = csp.get_all_variables()

    while len(queue) > 0:
        variable = queue.pop(0)
        deque.append(variable)

        reduced_variables = eliminate_from_neighbors(csp, variable)

        if reduced_variables == None:
            return None
        else:
            for var in reduced_variables:
                if not var in queue:
                    queue.append(var)

    return deque


# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?


csp = get_pokemon_problem()
domain_reduction(csp)
ANSWER_3 = solve_constraint_dfs(csp)[1]


def solve_constraint_propagate_reduced_domains(problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    if check_all_constraints(problem) == False or has_empty_domains(problem) == True:
        return (None, 1)
    elif check_all_constraints(problem) == True and len(problem.unassigned_vars) == 0:
        return (problem.assignments, 1)
    else:
        winner = (None, 0)
        unassigned_var = problem.pop_next_unassigned_var()
        domains = problem.get_domain(unassigned_var)

        count = 1
        for domain in domains:
            new_problem = problem.copy()
            new_game = new_problem.set_assignment(unassigned_var, domain)
            elim = eliminate_from_neighbors(new_game, unassigned_var)
            domain_reduction(new_game, elim)
            solution = solve_constraint_forward_checking(new_game)
            count = count + solution[1]
            if solution[0] != None:
                winner = solution
                break
        winner = winner[0], count
        return winner


# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?

ANSWER_4 = solve_constraint_propagate_reduced_domains(get_pokemon_problem())[1]


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    deque = [] 

    if queue == None:
         queue = csp.get_all_variables()

    while len(queue) > 0:
        variable = queue.pop(0)
        deque.append(variable)

        reduced_variables = eliminate_from_neighbors(csp, variable)

        if reduced_variables == None:
            return None
        else:
            for v in reduced_variables:
                if enqueue_condition_fn(csp, v):
                    queue = queue + [v]

    return deque


def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False""" 
    return True
    
def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    return True if len(csp.get_domain(var)) == 1 else False

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False

#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    if check_all_constraints(problem) == False or has_empty_domains(problem) == True:
        return (None, 1)
    elif check_all_constraints(problem) == True and len(problem.unassigned_vars) == 0:
        return (problem.assignments, 1)
    else:
        winner = (None, 0)
        unassigned_var = problem.pop_next_unassigned_var()
        domains = problem.get_domain(unassigned_var)

        count = 1
        for domain in domains:
            new_problem = problem.copy()
            new_game = new_problem.set_assignment(unassigned_var, domain)
            
            if enqueue_condition != None:
                propagate(enqueue_condition, new_game, [unassigned_var])

            solution = solve_constraint_generic(new_game, enqueue_condition)
            count = count + solution[1]
            if solution[0] != None:
                winner = solution
                break
        winner = winner[0], count
        return winner

# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)

ANSWER_5 = solve_constraint_generic(get_pokemon_problem(), condition_singleton)[1]


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    return True if m - n == 1 or m - n == -1 else False

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return False if m - n == 1 or m - n == -1 else True

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    combinations = []
    for index1, variable1 in enumerate(variables):
        for index2, variable2 in enumerate(variables):
            if index2 > index1:
                combinations.append(Constraint(variable1, variable2, constraint_different))
    return combinations



#### SURVEY ####################################################################

NAME = "Maximilian Deichmann"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
