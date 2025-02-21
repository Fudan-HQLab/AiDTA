import numpy as np
import copy
from config import CONFIG
from game import availabel,game_end
import uuid
import os
import subprocess

# list1 = ["AAA","CCC&GGG"]
# list2 = ["...","(((&)))"]
# fragment
list1 = ['GGC&GCC', 'CGT&ACG', 'TAT&ATA', 'CAC&GTG', 'AAG&CTT', 'TACA&TGTA', 'ATCG&CGAT', 'AATA&TATT', 'TAAA&TTTA', 'GTGG&CCAC', 'GGGTG&CACCC', 'CCAGC&GCTGG', 'CGGTG&CACCG', 'AGGTG&CACCT', 'TCAGG&CCTGA',  'TATCTG&CAGATA', 'AACATT&AATGTT', 'GACATT&AATGTC', 'GGGGCA&TGCCCC', 'CTGGCA&TGCCAG', 'AGG', 'CTG', 'GGG', 'GAA', 'GAC', 'GAGA', 'GCCA', 'ATTT', 'TCTG', 'GTTG', 'TTAGT', 'TTAGA', 'TTGGT', 'TTGGA', 'TTTGC', 'TTTTAA', 'TCTTTG', 'TTTTAC', 'TACGTC', 'TTCTGG']
# structure of fragment
list2 = ['(((&)))', '(((&)))', '(((&)))', '(((&)))', '(((&)))', '((((&))))', '((((&))))', '((((&))))', '((((&))))', '((((&))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '...', '...', '...', '...', '...', '....', '....', '....', '....', '....', '.....', '.....', '.....', '.....', '.....', '......', '......', '......', '......', '......']


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):

    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.p = prior_p

    def expand(self, act_probs):
        "Expand the tree by creating new nodes"
        for sequence, structure, prob in act_probs:
            sequence_structure = (sequence, structure)
            if sequence_structure not in self.children:
                self.children[sequence_structure] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        Select the node that provides the maximum Q + U from the child nodes
        return: a tuple (sequence_structure, next_node)
        """
        return max(self.children.items(),
                   key=lambda sequence_structure_node: sequence_structure_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """
        Compute and return the value of this node, which is a combination of Q and the prior of this node
        c_puct: controls the relative influence (0, inf)
        """
        self.u = (c_puct * self.p *
                  np.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.Q + self.u

    def update(self, leaf_value):
        """
        Update the node's value based on the evaluation from the leaf node
        leaf_value: the evaluation value of this child node from the perspective of the current player
        """
        # Count visits
        self.n_visits += 1
        # Update Q value, using incremental updates based on the average value from all visits
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits

        # Use recursion to update all the nodes (the subtree of the current node)

    def update_recursive(self, leaf_value):
        """Similar to calling update(), but updates all direct parent nodes"""
        # If it's not the root node, first update the parent node of this node
        if self.parent:
            self.parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """Check if it is a leaf node, i.e., a node that has not been expanded"""
        return self.children == {}

    def is_root(self):
        return self.parent is None



class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=10000, n_playout=1000):
        """policy_value_fn: A function that takes the board state and returns the move probabilities and board evaluation score"""
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, sequence, structure):
        """
        Perform a search and update the tree nodes' parameters in reverse order based on the leaf node's evaluation
        Note: the state is modified in-place, so a copy must be provided
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            """            
            Greedy algorithm to select the next action, the following 'action' takes only valid actions
            act_probs = zip(sequences, structures, act_probs[move])
            """

            sequence_structure, node = node.select(self._c_puct)

            sequence, structure = sequence_structure

        # Use the network to evaluate the leaf node, the network outputs a list of (action, probability) tuples p and the score from the current player's perspective [-1, 1]

        # First check if the game is over, then perform neural network evaluation
        if len(sequence.replace('&', '')) < 50:
            act_probs, leaf_value = self._policy(sequence, structure, list1, list2)
            node.expand(act_probs)
        else:
            leaf_value = game_end(sequence, structure)
            if leaf_value == 1:
                random_filename = str(uuid.uuid4())

                file = open(random_filename + ".txt", "w")
                file.write(sequence.replace('&', ''))
                file.close()

                # Call RNAstructure's Fold tool
                subprocess.run(['Fold', random_filename + ".txt", random_filename + ".ct"])

                # Call RNAstructure's ct2dot tool
                subprocess.run(['ct2dot', random_filename + ".ct", '1', random_filename + ".dbn"])

                with open(random_filename + ".dbn", 'r') as file:
                    lines = file.readlines()
                    if len(lines) >= 3:  # Ensure the file has at least 3 lines
                        # Print the third line and use strip() to remove trailing newline
                        pre_sec = lines[2].strip()
                    else:
                        print("The file has insufficient number of lines")

                # Delete the generated files
                os.remove(random_filename + ".txt")
                os.remove(random_filename + ".ct")
                os.remove(random_filename + ".dbn")

                with open("best_sequence.txt",'a+') as file:
                    file.seek(0)
                    lines = file.readlines()
                    if all(sequence.replace('&', '') != line.split(' ')[0] for line in lines):
                        file.write(sequence.replace('&', '') + " " + pre_sec + "\n")
                        file.close()

        node.update_recursive(leaf_value)

    def get_move_probs(self, sequence, structure, temp=1e-3):
        """
        Run all simulations and return the available actions with their respective probabilities
        state: Current game state
        temp: Temperature parameter between (0, 1]
        """
        for n in range(self._n_playout):
            sequence_copy = copy.deepcopy(sequence)
            structure_copy = copy.deepcopy(structure)

            self._playout(sequence_copy, structure_copy)

        # Calculate the move probabilities based on the visit count at the root node
        act_visits = [(sequence_structure, node.n_visits)
                      for sequence_structure, node in self._root.children.items()]
        sequence_structures, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return sequence_structures, act_probs

    def update_with_move(self, sequence_structure):
        """
        Move one step forward on the current tree, keeping everything we already know about the subtree
        """
        if sequence_structure in self._root.children:
            self._root = self._root.children[sequence_structure]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return 'MCTS'



# AI Player based on MCTS
class MCTSPlayer(object):

    def __init__(self, policy_value_function, c_puct=5, n_playout=1000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_ind(self, p):
        self.player = p

    # Reset the search tree
    def reset_player(self):
        self.mcts.update_with_move(-1)

    def __str__(self):
        return 'MCTS {}'.format(self.player)

    # Get action
    def get_action(self, sequence, structure):
        # Use the pi vector returned by the MCTS algorithm, like in the AlphaGo_Zero paper
        sequence_structure_probs = np.zeros(2000)
        moves, sequences, structures = availabel(sequence, structure, list1, list2)
        # print("Moves in get_action", moves)

        sequence_structures, act_probs = self.mcts.get_move_probs(sequence, structure, 1e-3)
        sequence_structure_probs[list(moves)] = act_probs
        # print("sequence_structures", sequence_structures)
        # print("act_probs", act_probs)

        # Add Dirichlet noise for exploration (needed in self-play)
        # Create an empty dictionary
        mapping = {}

        # Use the zip function to pair elements from moves and sequence_structures
        for move, sequence_structure in zip(moves, sequence_structures):
            mapping[move] = sequence_structure

        # Generate the move
        move = np.random.choice(
            moves,
            p=0.65 * act_probs + 0.35 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(act_probs)))
        )

        sequence_structure = mapping[move]
        # Update the root node and reuse the search tree
        self.mcts.update_with_move(sequence_structure)

        return sequence_structure, sequence_structure_probs




