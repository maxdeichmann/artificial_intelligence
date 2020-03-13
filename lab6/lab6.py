# MIT 6.034 Lab 6: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2, 1]

nn_cross = [2, 2, 1]

nn_stripe = [3, 1]

nn_hexagon = [6, 1]

nn_grid = [4, 2, 1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    return 1 if x >= threshold else 0

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1+e**(-steepness*(x-midpoint)))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(0,x)

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return (-1/2)*((desired_output-actual_output)**2)


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""

    sorted_neurons = net.topological_sort()
    output_values = {}

    for neuron in sorted_neurons:
        output_values[neuron] = 0
        incoming_neurons = net.get_incoming_neighbors(neuron)
        input_sum = 0
        for incoming_neuron in incoming_neurons:
            value = node_value(incoming_neuron, input_values, output_values)
            wire = net.get_wire(incoming_neuron, neuron)
            input_sum += wire.get_weight()*value
        output_values[neuron] = threshold_fn(input_sum)
    
    final_key = list(output_values.keys())[-1]
    return output_values[final_key], output_values



#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""

    output = -10000000
    combination = []

    ranges = []
    for value in inputs:
        ranges.append([value-step_size, value, value+step_size])

    for inputOne in ranges[0]:
        for inputTwo in ranges[1]:
            for inputThree in ranges[2]:
                value = func(inputOne, inputTwo, inputThree)
                if value > output:
                    output = value
                    combination = [inputOne, inputTwo, inputThree]
    return output, combination

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""

    output = set([wire, wire.startNode, wire.endNode])

    neighbors = net.get_outgoing_neighbors(wire.endNode)
    for neighbor in neighbors:
        new_wires = net.get_wires(wire.endNode, neighbor)
        for new_wire in new_wires:
            results = get_back_prop_dependencies(net, new_wire)
            for result in results:
                output.add(result)

    return output


def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """

    deltas = {}
    sorted_neurons = net.topological_sort()
    sorted_neurons.reverse()

    for neuron in sorted_neurons:
        neuron_output = neuron_outputs[neuron]

        if net.is_output_neuron(neuron):
            neuron_delta = (desired_output-neuron_output)*neuron_output*(1-neuron_output)
            deltas[neuron] = neuron_delta
        else:
            delta_sum = 0
            for neighbor in net.get_outgoing_neighbors(neuron):
                wires = net.get_wires(neuron,neighbor)
                for wire in wires:
                    delta_sum += wire.weight * deltas[wire.endNode]
            deltas[neuron] = delta_sum * neuron_output* (1-neuron_output)

    return deltas

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""

    neuron_deltas = calculate_deltas(net, desired_output, neuron_outputs)

    for neuron in neuron_deltas:
         incoming_neurons = net.get_incoming_neighbors(neuron)
         for incoming_neuron in incoming_neurons:
            wires = net.get_wires(incoming_neuron, neuron)
            for wire in wires:
                
                new_weight = wire.weight + r * neuron_deltas[neuron] * node_value(incoming_neuron, input_values, neuron_outputs)
                wire.set_weight(new_weight)
    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""

    current_net_output, current_node_values = forward_prop(net, input_values, sigmoid)
    acc = accuracy(desired_output, current_net_output)

    counter = 0
    while acc <= minimum_accuracy:

        current_net_output, current_node_values = forward_prop(net, input_values, sigmoid)
        net = update_weights(net, input_values, desired_output, current_node_values, r)
        updated_net_output, updated_node_values = forward_prop(net, input_values, sigmoid)
        acc = accuracy(desired_output, updated_net_output)
        counter += 1

    return net, counter

#### Part 5: Training a Neural Net #############################################

ANSWER_1 = 50
ANSWER_2 = 50
ANSWER_3 = 3
ANSWER_4 = 70
ANSWER_5 = 15

ANSWER_6 = 1
ANSWER_7 = "checkerboard"
ANSWER_8 = ['small', 'medium', 'large']
ANSWER_9 = 'B'

ANSWER_10 = 'D'
ANSWER_11 = 'AC'
ANSWER_12 = 'AE'


#### SURVEY ####################################################################

NAME = "Maximilian Deichmann"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 8
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
