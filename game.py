import numpy as np
import uuid
import os
import copy
import time
from collections import deque
import torch
import random
# Initial sequence and secondary structure
import subprocess
# Nucleic acid sequence and corresponding secondary structure
# list1 = ["AAA","CCC&GGG"]
# list2 = ["...","(((&)))"]
# fragment
list1 = ['GGC&GCC', 'CGT&ACG', 'TAT&ATA', 'CAC&GTG', 'AAG&CTT', 'TACA&TGTA', 'ATCG&CGAT', 'AATA&TATT', 'TAAA&TTTA', 'GTGG&CCAC', 'GGGTG&CACCC', 'CCAGC&GCTGG', 'CGGTG&CACCG', 'AGGTG&CAC    CT', 'TCAGG&CCTGA',  'TATCTG&CAGATA', 'AACATT&AATGTT', 'GACATT&AATGTC', 'GGGGCA&TGCCCC', 'CTGGCA&TGCCAG', 'AGG', 'CTG', 'GGG', 'GAA', 'GAC', 'GAGA', 'GCCA', 'ATTT', 'TCTG', 'GTTG', 'TT    AGT', 'TTAGA', 'TTGGT', 'TTGGA', 'TTTGC', 'TTTTAA', 'TCTTTG', 'TTTTAC', 'TACGTC', 'TTCTGG']
# structure of fragment
list2 = ['(((&)))', '(((&)))', '(((&)))', '(((&)))', '(((&)))', '((((&))))', '((((&))))', '((((&))))', '((((&))))', '((((&))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '...', '...', '...', '...', '...', '....', '....', '....', '....', '....', '.....', '.....', '.....', '.....', '.....', '......', '......', '......', '......', '......']

src_vocab = {'A':1,'G':2,'C':3,'T':4}
sec_vocab = {'(':5,')':5,'.':6}

def availabel(sequence, structure, list1, list2):
    sequences = []
    structures = []
    results = []  # Stores all insert results for this round
    seq_dict = {i: seq for i, seq in enumerate(list1)}
    # Go through all possible insertion sequences
    move = []
    for i, (insert_sequence, insert_structure) in enumerate(zip(list1, list2)):
        # Go through all possible insertion positions
        positions = [k for k, char in enumerate(sequence) if char == '&']
        positions.append(len(sequence))
        positions = [pos - sequence[:pos].count('&') for pos in positions]  # does not count towards the length of the "&" symbol
        seq_parts = sequence.split("&")
        struc_parts = structure.split("&")
        for insert_index,position in zip(range(len(seq_parts)),positions):
            # Create new copies of sequences and structures for insertion
            new_seq_parts = seq_parts.copy()
            new_struc_parts = struc_parts.copy()
            # Inserts sequences and structures at specified locations
            new_seq_parts[insert_index] = new_seq_parts[insert_index] + insert_sequence
            new_struc_parts[insert_index] = new_struc_parts[insert_index] + insert_structure
            # Reassemble sequences and structures
            new_sequence = "&".join(new_seq_parts)
            new_structure = "&".join(new_struc_parts)
            sequences.append(new_sequence)
            structures.append(new_structure)
            # Adds the insert result to the result list
            results.append((new_sequence, new_structure))
            #  Combines strings into a numpy array
            combined = 50*i + position
            move.append(combined)
    # print("move",move)
    # print("results",results)
    return move,sequences,structures

# Initial sequence and structure
sequence = ""
structure = ""


def State(sequence, structure):
    max_len = 61
    # Perform insertion until the sequence length reaches 50

    encoded_seq = np.zeros((4, max_len))
    encoded_sec = np.zeros((2, max_len))

    for i, nuc in enumerate(sequence.replace('&', '')):
        # Convert nucleotides to indices based on the vocabulary dictionary and add to the encoded sequence
        encoded_seq[src_vocab[nuc]-1, i] = src_vocab[nuc]

    for i, sec in enumerate(structure.replace('&', '')):
        # Convert nucleotides to indices based on the vocabulary dictionary and add to the encoded structure
        encoded_sec[sec_vocab[sec]-5, i] = sec_vocab[sec]

    # Combine the two matrices into a single 6*61 matrix
    combined_matrix = np.vstack((encoded_seq, encoded_sec))

    combined_matrix = np.expand_dims(combined_matrix, axis=0)  # Add a dimension at the first axis

    return combined_matrix




def game_end(sequence, structure):
    # Use uuid to generate a random filename
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

    diff = 0
    pre_sec = pre_sec.replace(".", "0").replace("(", "1").replace(")", "1")
    structure1 = structure.replace('&', '').replace(".", "0").replace("(", "1").replace(")", "1")

    for i in range(len(pre_sec)):
        diff += abs(int(pre_sec[i]) - int(structure1[i]))

    similarity_score = 1 - diff / len(pre_sec)

    print("similarity_score", similarity_score)
    print(lines[2].strip())
    print(structure.replace('&', ''))
    # print(len(sequence.replace('&', '')))  # Do not count the "&" symbols' length
    # print("The sequence has exceeded 50")

    if similarity_score >= 0.9:
        return 1
    else:
        return -1


def self_play(sequence, structure, player):

    sequences, structures, mcts_probs,  = [], [], []
    play_times =0
    while True:
        if len(sequence.replace('&', '')) < 50:

            sequence_structure, sequence_structure_probs = player.get_action(
                                                sequence,
                                                structure)
            play_times += 1
            sequences.append(sequence)
            structures.append(structure)
            mcts_probs.append(sequence_structure_probs)
            sequence, structure = sequence_structure


        else:

            score = game_end(sequence,structure)
            scores = np.ones(play_times)*score
            if score == 1:
                with open("best_sequence.txt",'a') as file:
                    file.write(sequence.replace('&', '') + " " + structure.replace('&', '') + "\n")
                    file.close()


            return zip(sequences, structures, mcts_probs, scores)

