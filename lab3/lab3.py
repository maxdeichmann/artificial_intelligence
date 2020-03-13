# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""

    for chain in board.get_all_chains():
        if len(chain) > 3:
            return True
    if board.count_pieces() >= board.num_rows * board.num_cols:
        return True
    return False

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    possible_boards = []
    if is_game_over_connectfour(board) == False:
        for col in range(0,board.num_cols):
            if board.is_column_full(col) == False:
                new_board = board.add_piece(col)
                possible_boards.append(new_board)
    return possible_boards

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""

    for chain in board.get_all_chains():
        if len(chain) > 3:
            if is_current_player_maximizer:
                return -1000
            else:
                return 1000
    if board.count_pieces() >= board.num_rows * board.num_cols:
        return 0

def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""

    endgame_score = endgame_score_connectfour(board, is_current_player_maximizer)
    amount_of_free_places = board.num_rows * board.num_cols - board.count_pieces()
    premium = amount_of_free_places * 100

    returned_value = abs(endgame_score) + premium
    return -returned_value if is_current_player_maximizer else returned_value

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""

    current_player_score = score_calculator(board.get_all_chains(is_current_player_maximizer))
    other_player_score = score_calculator(board.get_all_chains(not is_current_player_maximizer))

    return current_player_score - other_player_score


def score_calculator(chains):
    score = 0
    for chain in chains:
        score = score + len(chain)**3
    return score


# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""

    final = []
    if state.is_game_over():
        return ([state], state.get_endgame_score(False), 1)
    else:
        stack = [[state]]
        
        while len(stack) > 0:

            first_element = stack.pop(0)

            if first_element[-1].is_game_over():
                final.append(first_element)
            else:

                new_possibilities = first_element[-1].generate_next_states()
                for possibility in new_possibilities:

                    new_path = first_element + [possibility]
                    stack = [new_path] + stack


        scores = [path[-1].get_endgame_score(False) for path in final]
        max_score = max(scores)
        index = scores.index(max_score)
        return (final[index], final[index][-1].get_endgame_score(False), len(scores))




# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))
# pretty_print_dfs_type(dfs_maximizing(state_NEARLY_OVER))


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""

    final = []
    if state.is_game_over():
        return ([state], state.get_endgame_score(maximize), 1)
    else:
        stack = [[state]]
        
        while len(stack) > 0:

            first_element = stack.pop(0)

            if first_element[-1].is_game_over():
                final.append(first_element)
            else:

                new_possibilities = first_element[-1].generate_next_states()
                for possibility in new_possibilities:

                    new_path = first_element + [possibility]
                    stack = [new_path] + stack

    score = 0
    winners = []
    index = 0
    while len(final) > 0:
        if len(final) > index:
            eternal_copy = final[index].copy()
            firstElementReduced = final[index].copy()
            firstElementReduced.pop()
            group = []
            for element in final:
                elementReduced  = element.copy()
                elementReduced.pop()
                if len(element) == len(eternal_copy):
                    if all(True if element == elementReduced[ind] else False for ind, element in enumerate(firstElementReduced)) == True:
                        group.append(element)


            final = [x for x in final if x not in group]

            for element in group:
                if not isinstance(element[-1], int):
                    score += 1
                    element[-1] = element[-1].get_endgame_score(not maximize)

            group_copy = group.copy()
            value = min(group_copy, key=lambda x: x[-1]) if len(eternal_copy) % 2 == maximize else max(group_copy, key=lambda x: x[-1])
            winners.append(value.copy())
            for element in group:
                if element[-1] == value[-1]:
                    if len(element) > 1:
                        element.pop(-2)
                        final.insert(index, element)
                    else:
                        # find winner
                        winners = [x for x in winners if x[-1] == element[0]]
                        for i,possibility in enumerate(winners[0][-2].generate_next_states()):
                            if possibility.get_endgame_score(not maximize) == element[0]:
                                winners[0][-1] = possibility

                        return (winners[0], element[0], score)
            index += 1
        else:
            index = 0



# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(GAME1))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""

    winning_path = (None, -10000, 0) if maximize == True else (None, 10000, 0)
    evaluations = 0
    
    if state.is_game_over():
        return [state], state.get_endgame_score(maximize), 1
    elif depth_limit == 0:
        return [state], heuristic_fn(state.snapshot, maximize), 1
    else:
        for child_state in state.generate_next_states():
            evaluated_grand_child = minimax_search(child_state, heuristic_fn, (depth_limit - 1), not maximize)
            evaluations += evaluated_grand_child[2]
            if evaluated_grand_child[1] > winning_path[1] if maximize == True else evaluated_grand_child[1] < winning_path[1]:
                winning_path = evaluated_grand_child
        return [state] + winning_path[0], winning_path[1], evaluations


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    winning_path = ([state], alpha, 0) if maximize == True else ([state], beta, 0)
    evaluations = 0
    
    if state.is_game_over():
        if alpha >= beta:
            return [state], alpha, 0
        else:
            return [state], state.get_endgame_score(maximize), 1
    elif depth_limit == 0:
        if alpha >= beta:
            return [state], alpha, 0
        else:
            return [state], heuristic_fn(state.snapshot, maximize), 1
    else:
        for child_state in state.generate_next_states():
            evaluated_grand_child = []
            if maximize:
                evaluated_grand_child = minimax_search_alphabeta(child_state, winning_path[1], beta, heuristic_fn, (depth_limit - 1), False)
            else:
                evaluated_grand_child = minimax_search_alphabeta(child_state, alpha, winning_path[1], heuristic_fn, (depth_limit - 1), True)

            evaluations += evaluated_grand_child[2]
            if evaluated_grand_child[1] > winning_path[1] if maximize == True else evaluated_grand_child[1] < winning_path[1]:
                winning_path = evaluated_grand_child
        return [state] + winning_path[0], winning_path[1], evaluations
        


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    raise NotImplementedError
    print(depth_limit)
    anytime_value = AnytimeValue()
    while depth_limit > 0:
        (path, score, evaluations) = minimax_search_alphabeta(state, -INF, INF, heuristic_fn, 10, not maximize)
        print(score)
        state = path
        depth_limit = depth_limit - 10



# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = "Maximilian Deichmann"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 4
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
