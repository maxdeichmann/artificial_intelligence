# MIT 6.034 Lab 1: Rule-Based Systems
# Written by 6.034 staff

from production import IF, AND, OR, NOT, THEN, DELETE, forward_chain, pretty_goal_tree
from data import *
import pprint

pp = pprint.PrettyPrinter(indent=1)
pprint = pp.pprint

#### Part 1: Multiple Choice #########################################

ANSWER_1 = '2'

ANSWER_2 = '4'

ANSWER_3 = '2'

ANSWER_4 = '0'

ANSWER_5 = '3'

ANSWER_6 = '1'

ANSWER_7 = '0'

#### Part 2: Transitive Rule #########################################

# Fill this in with your rule 
transitive_rule = IF(AND('(?x) beats (?y)', '(?y) beats (?z)'), THEN("(?x) beats (?z)") )

poker_data = [ 'two-pair beats pair',
               'three-of-a-kind beats two-pair',
               'straight beats three-of-a-kind',
               'flush beats straight',
               'full-house beats flush',
               'straight-flush beats full-house' ]

# recursivelyAddRules(poker_data)

# def recursivelyAddRules(poker_data):
# for rule in poker_data:
#     for potential_predecessor in poker_data:
#         p_pre_li = potential_predecessor.split(' ')
#         rule_li = potential_predecessor.split(' ')
#         if rule_li[2] == p_pre_li[2]:

# You can test your rule by uncommenting these pretty print statements
#  and observing the results printed to your screen after executing lab1.py
# pprint(forward_chain([transitive_rule], abc_data))
# pprint(forward_chain([transitive_rule], poker_data))
# pprint(forward_chain([transitive_rule], minecraft_data))


#### Part 3: Family Relations #########################################

# Define your rules here. We've given you an example rule whose lead you can follow:
friend_rule = IF( AND("person (?x)", "person (?y)"), THEN ("friend (?x) (?y)", "friend (?y) (?x)") )

self_rule = IF( AND("person (?x)"), THEN ("self (?x) (?x)") )
sibling_rule = IF(AND('parent (?a) (?x)', 'parent (?a) (?y)', NOT('self (?y) (?x)')), THEN('sibling (?x) (?y)', 'sibling (?y) (?x)'))
child_rule = IF( AND("person (?x)", "person (?y)", "parent (?x) (?y)"), THEN ("child (?y) (?x)") )
cousin_rule = IF(AND('sibling (?x) (?y)', 'parent (?x) (?a)', 'parent (?y) (?b)', NOT('siblings (?a) (?b)')), THEN('cousin (?a) (?b)', 'cousin (?b) (?a)'))
grandparent_rule = IF( AND("person (?x)", "person (?y)", "parent (?x) (?a)", "parent (?a) (?y)"), THEN ("grandparent (?x) (?y)", "grandchild (?y) (?x)"))

# Add your rules to this list:
family_rules = [self_rule, sibling_rule, child_rule, cousin_rule, grandparent_rule]

# Uncomment this to test your data on the Simpsons family:

# These smaller datasets might be helpful for debugging:
# pprint(forward_chain(family_rules, sibling_test_data, verbose=True))
# pprint(forward_chain(family_rules, grandparent_test_data, verbose=True))

# The following should generate 14 cousin relationships, representing 7 pairs
# of people who are cousins:
# black_family_cousins = [
#     relation for relation in
#     forward_chain(family_rules, black_data, verbose=False)
#     if "cousin" in relation ]

# To see if you found them all, uncomment this line:
# pprint(black_family_cousins)


#### Part 4: Backward Chaining #########################################

# Import additional methods for backchaining
from production import PASS, FAIL, match, populate, simplify, variables

def backchain_to_goal_tree(rules, hypothesis):
    """
    Takes a hypothesis (string) and a list of rules (list
    of IF objects), returning an AND/OR tree representing the
    backchain of possible statements we may need to test
    to determine if this hypothesis is reachable or not.

    This method should return an AND/OR tree, that is, an
    AND or OR object, whose constituents are the subgoals that
    need to be tested. The leaves of this tree should be strings
    (possibly with unbound variables), *not* AND or OR objects.
    Make sure to use simplify(...) to flatten trees where appropriate.
    """
    print(rules)
    hy_li = hypothesis.split(' ')
    protagonist = hy_li.pop(0)

    if len(rules) == 0:
        return hypothesis
    else:
        return simplify(backchain_to_goal_tree_rec(rules, hypothesis, protagonist))


def backchain_to_goal_tree_rec(rules, hypothesis, protagonist):
    hy_li = hypothesis.split(' ')
    hy_li.pop(0)
    clean_hypothesis = ' '.join(hy_li)
    goaltree = OR()
    goaltree.append(protagonist+" "+clean_hypothesis)

    for rule in rules:
        clean_consequent = removeX(rule.consequent())
        if clean_consequent == clean_hypothesis:

            antecedents = rule.antecedent()
            if isinstance(antecedents, list):
                subtree = []
                if isinstance(antecedents, OR):
                    subtree = OR()
                if isinstance(antecedents, AND):
                    subtree = AND()
                for antecedent in antecedents:
                    subtree.append(backchain_to_goal_tree_rec(rules, antecedent, protagonist))
                
                goaltree.append(subtree)
            else:
                goaltree.append(backchain_to_goal_tree_rec(rules, antecedents, protagonist))
    return goaltree

def removeX(rule):
    return rule[5:]


# Uncomment this to test out your backward chainer:
pretty_goal_tree(backchain_to_goal_tree(zookeeper_rules, 'opus is a penguin'))


#### Survey #########################################

NAME = "Maximilian Deichmann"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = 4
WHAT_I_FOUND_INTERESTING = "None"
WHAT_I_FOUND_BORING = "None"
SUGGESTIONS = "None"


###########################################################
### Ignore everything below this line; for testing only ###
###########################################################

# The following lines are used in the tester. DO NOT CHANGE!
print("(Doing forward chaining. This may take a minute.)")
transitive_rule_poker = forward_chain([transitive_rule], poker_data)
transitive_rule_abc = forward_chain([transitive_rule], abc_data)
transitive_rule_minecraft = forward_chain([transitive_rule], minecraft_data)
family_rules_simpsons = forward_chain(family_rules, simpsons_data)
family_rules_black = forward_chain(family_rules, black_data)
family_rules_sibling = forward_chain(family_rules, sibling_test_data)
family_rules_grandparent = forward_chain(family_rules, grandparent_test_data)
family_rules_anonymous_family = forward_chain(family_rules, anonymous_family_test_data)
