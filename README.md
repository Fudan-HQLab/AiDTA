# AiDTA-RBD

**AiDTA** (**AI**-enabled **D**ocking-**T**hen-**A**ssembling) is a deep reinforcement-learning framework for the de novo design of protein-targeting nucleic acid aptamers. This repository provides the implementation configured for the receptor-binding domain (RBD) of the SARS-CoV-2 spike protein.

Conventional aptamer discovery generally relies on SELEX screening of large random sequence libraries. AiDTA instead assembles and optimizes candidate aptamers in silico using a target-specific pool of docked nucleic acid fragments, Monte Carlo tree search (MCTS), a residual policy-value network, and secondary-structure validation.

## Method

```text
Nucleic acid fragment generation
              |
              v
3D-structure prediction with 3dRNA/DNA
              |
              v
Fragment-RBD docking with HDOCK
              |
              v
Construction of an RBD-specific fragment pool
              |
              v
MCTS-guided fragment assembly
              |                         |
              v                         v
RNAstructure validation          Self-play replay buffer
              |                         |
              v                         v
Candidate aptamers               Policy-value network training
```

Single-stranded fragments of 3–6 nucleotides and paired fragments of 3–6 base pairs are first modeled and then docked against the SARS-CoV-2 RBD. Fragments are filtered according to their docking sites and docking scores. The action space also includes four single-nucleotide extension actions to increase assembly flexibility.

The filtered fragment sequences and their dot-bracket secondary structures are stored in `CONFIG["list1"]` and `CONFIG["list2"]`, respectively.

## RBD-specific design parameters

The default parameters in `config.py` define the search and training spaces used for RBD-targeting aptamer design.

### Fragment and sequence space

| Parameter | Value | Description |
| --- | ---: | --- |
| Target | SARS-CoV-2 spike RBD | Protein target used to construct the fragment pool. |
| Fragment actions | 44 | Four single-nucleotide actions, 20 single-stranded fragments, and 20 paired fragments. |
| `min_len` | 50 nt | Minimum candidate length at which evaluation begins. |
| `max_len` | 61 nt | Maximum encoded length, allowing the longest fragment to be inserted from a 49-nt state. |
| `policy_num` | 2,200 | Policy-output dimension: 44 fragment actions × 50 insertion positions. |
| Acceptance threshold | 0.90 | Minimum similarity between paired/unpaired secondary-structure states required for a positive reward. |

### MCTS parameters

| Parameter | Value | Description |
| --- | ---: | --- |
| `play_out` | 100 | Number of MCTS simulations at each assembly step. |
| `c_puct` | 10 | Exploration weight in the PUCT selection rule. |
| `dirichlet` | 0.2 | Concentration parameter of the Dirichlet distribution used for self-play exploration. |
| Action-probability mixture | 0.65 / 0.35 | Mixture of network-predicted probabilities and Dirichlet noise during self-play. |

### Training parameters

| Parameter | Value | Description |
| --- | ---: | --- |
| `buffer_size` | 10,000 | Maximum number of self-play records retained in the replay buffer. |
| `batch_size` | 512 | Number of records sampled for each network update. |
| `epochs` | 5 | Maximum number of optimization epochs per mini-batch. |
| `kl_targ` | 0.02 | Target KL divergence for adaptive learning-rate control. |
| `game_batch_num` | 3,000 | Number of training-loop iterations. |
| Initial learning rate | 0.001 | Adam learning rate before adaptive scaling. |
| L2 regularization | 0.002 | Weight-decay coefficient used by the optimizer. |

### Network parameters

| Parameter | Value |
| --- | ---: |
| Input representation | 6 × 61 |
| Residual channels | 256 |
| Residual blocks | 7 |
| Policy-output dimension | 2,200 |
| Value-output dimension | 1 |
| Default device | `cuda:0` |

## Repository contents

| Path | Purpose |
| --- | --- |
| `config.py` | Stores the RBD fragment pool, sequence-length limits, MCTS parameters, training parameters, and model-file paths. |
| `game.py` | Implements fragment insertion, state encoding, RNAstructure-based evaluation, and self-play. |
| `mcts.py` | Implements MCTS traversal, simulation, action selection, and exploration. |
| `cnn_net.py` | Implements the one-dimensional residual policy-value network and its inference and training interfaces. |
| `collect.py` | Continuously collects self-play data and saves the replay buffer. |
| `train.py` | Trains the policy-value network from the replay buffer. |
| `main_worker.py` | Launches collection and training processes concurrently and manages their logs. |
| `unique.py` | Post-processing utility for deduplicating candidate sequences. |
| `fragment/hdock.zip` | Complete precomputed nucleic acid fragment library containing 10,880 PDB structures. Extract and reuse these fragments as HDOCK ligands for a new protein target; no repeat 3D fragment prediction is required. |
| `data_avalibility/` | Example outputs from one target-specific AiDTA run, including candidate sequences and HDOCK/HADDOCK score tables. |
| `current_policy.pkl` | Trained PyTorch model checkpoint, written by `train.py` at runtime. Not distributed; see [Model weights are not distributed](#model-weights-are-not-distributed). |
| `train_data_buffer.pkl` | Serialized self-play replay buffer generated at runtime. |
| `best_sequence.txt` | Sequence-structure pairs generated and accepted during self-play. |
| `unique_best_sequence.txt` | Deduplicated and named candidate sequence-structure pairs. |

### Example data scope

The supplied text files contain 20,006 sequence/structure rows, 15,991 HDOCK rows, and 101 paired HDOCK/HADDOCK rows. The paired-score table has 101 distinct candidate IDs: 80 also occur in the HDOCK table, with identical HDOCK scores, and 21 are absent from that table. These files should therefore not be interpreted as a complete, documented top-100 filtering chain.

The sequence file has no explicit candidate-ID column. Mapping `aptN` to row N should be confirmed from the original run records before joining it to the docking tables. The original data files are distributed unchanged; run provenance and a complete ID mapping are not supplied here. Docking scores are computational outputs, not experimental affinity measurements.

### Model weights are not distributed

This repository distributes the source code, the precomputed fragment library, and the example outputs in `data_avalibility/` (for RBD 1.1.529: 20,006 candidate sequences with their secondary structures, 15,991 HDOCK docking records, and 101 rows of paired HDOCK/HADDOCK scores). It does **not** distribute the trained network weights (`current_policy.pkl`) or the replay buffer (`train_data_buffer.pkl`).

The checkpoints and replay buffers used to generate the example outputs are not included in this release. The source code can be used to train new models and generate new candidate sets, but the released files alone do not enable exact regeneration of the example outputs. Stochastic self-play is not a substitute for archiving the checkpoints, random seeds, and run metadata needed for reproducibility.

To generate weights and sequences for RBD (1.1.529), run the main launcher `main_worker.py`:

```bash
python main_worker.py --multiple-script collect.py --count 4 --specific-script train.py --timeout 7200
python unique.py
```

`train.py` writes the weights to `current_policy.pkl` (as set by `CONFIG["pytorch_model_path"]`), `collect.py` appends accepted sequence-structure pairs to `best_sequence.txt`, and `unique.py` produces the deduplicated `unique_best_sequence.txt`. Dirichlet noise is sampled at each self-play assembly decision in `MCTSPlayer.get_action`. These files can differ between runs; statistical equivalence to the released example outputs has not been established. Adjust `--timeout` (seconds) and `--count` to your hardware; the collectors keep running until the launcher is interrupted with `Ctrl+C`.

## Requirements

- A Python environment compatible with the selected PyTorch version
- A CUDA-capable GPU; the model uses `cuda:0` by default
- Python packages:
  - `numpy`
  - `torch`
  - `pandas` (required by `unique.py`)
- RNAstructure command-line programs available through the system `PATH`:
  - `Fold`
  - `ct2dot`

The repository already includes the complete precomputed fragment-structure library in `fragment/hdock.zip`. Therefore, adapting AiDTA to another protein target does not require repeating fragment generation or 3D-structure prediction with 3dRNA/DNA. HDOCK is still required to dock the supplied fragments against each new target. 3dRNA/DNA is needed only if the supplied fragment library is intentionally extended or replaced. These programs are not called directly during self-play or network training.

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv

# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install numpy pandas torch
```

Install the PyTorch build that matches your local CUDA environment.

Verify that RNAstructure is available:

```bash
Fold --help
ct2dot --help
```

## Configuration

The included `CONFIG` is set up for the supplied SARS-CoV-2 RBD fragment pool and model dimensions.

- `list1[i]` is a fragment sequence, and `list2[i]` is its corresponding dot-bracket structure.
- `list1` and `list2` must have the same length and remain in one-to-one correspondence.
- The `&` symbol separates paired strands or assembled fragments.
- Model and replay-buffer paths are controlled by `pytorch_model_path` and `train_data_buffer_path`, respectively.
- `policy_num` must equal `len(list1) * min_len`.
- `max_len` must accommodate the longest sequence produced by the final insertion operation.

The fully connected layers in `cnn_net.py` are dimensioned for `max_len = 61` and `policy_num = 2200`. If either parameter is changed, the dimensions of these network layers must be adjusted accordingly.

## Running self-play and training

### Two-terminal workflow

Start the self-play collection process:

```bash
python collect.py
```

Start the training process in a second terminal:

```bash
python train.py
```

The collector runs continuously until interrupted manually. The trainer runs for up to `CONFIG["game_batch_num"]` iterations (3,000 by default), but can wait indefinitely for a readable replay buffer. The collector repeatedly loads the latest model checkpoint, performs MCTS-guided fragment assembly, evaluates completed sequences, and appends self-play records to `train_data_buffer.pkl`. The trainer samples the replay buffer and saves the optimized network parameters to `current_policy.pkl`.

### Multiprocess launcher

`main_worker.py` can launch multiple collectors together with one time-limited trainer:

```bash
python main_worker.py \
  --multiple-script collect.py \
  --count 4 \
  --specific-script train.py \
  --timeout 7200
```

`--timeout` is specified in seconds. Output from each process is written to a separate log file under `logs/`. Collection processes continue to run until the entire task is stopped with `Ctrl+C`.

Display all launcher options:

```bash
python main_worker.py --help
```

## Candidate-sequence evaluation

When an assembled sequence reaches 50 nucleotides, `game.py` performs the following steps:

1. Writes the assembled sequence to a temporary file.
2. Uses RNAstructure `Fold` to predict its secondary structure.
3. Uses `ct2dot` to convert the CT result to dot-bracket notation.
4. Compares the paired and unpaired states in the predicted structure with those in the assembly target structure.
5. Assigns a positive reward when their similarity is at least 0.90.

The temporary files produced by RNAstructure are deleted after evaluation.

## Output files

Each record in `best_sequence.txt` contains a sequence and a dot-bracket structure. `game_end` writes the RNAstructure prediction, whereas a successful final self-play trajectory also writes the assembly target structure. The file can therefore contain both structure types; the current output format does not label their provenance:

```text
<sequence> <dot-bracket structure>
```

Similarity scores, predicted structures, and assembly target structures are also written to the collection logs. For successful self-play trajectories, the logs can contain both the RNAstructure prediction and the target structure assembled from the RBD fragment pool.

Run the post-processing utility to remove duplicate sequence-structure pairs:

```bash
python unique.py
```

`unique_best_sequence.txt` uses the following format:

```text
<candidate ID> <sequence> <dot-bracket structure>
```

The default candidate-name prefix is `Aptamer`. It can be changed through the `name` variable in `unique.py`.

## Inference

`PolicyValueNet.policy_value` accepts an encoded state rather than raw sequence and structure strings. Use `game.State` to prepare the model input:

```python
from cnn_net import PolicyValueNet
from config import CONFIG
from game import State

sequence = "AAACCCAAACCCCCCAAAAAA&GGG&GGGAAACCCCCC&GGG&GGG&GGGAAA"
structure = "...(((...((((((......&)))&)))...((((((&)))&)))&)))..."

model = PolicyValueNet(
    model_file=CONFIG["pytorch_model_path"],
    device="cuda:0",
)
state = State(sequence, structure)
action_probabilities, value = model.policy_value(state)

print("Action probabilities:", action_probabilities)
print("State value:", value)
```

After removing `&`, the sequence and structure must have exactly the same length, and the encoded length must not exceed 61.

## Adapting AiDTA to another protein target

`fragment/hdock.zip` contains the complete precomputed fragment library used by this workflow. Extract and reuse these fragment structures for a new target; do not regenerate the fragments or repeat 3D-structure prediction unless the library itself is being changed.

1. Extract the supplied fragment structures from `fragment/hdock.zip`.
2. Prepare the three-dimensional structure of the new protein target.
3. Dock the supplied fragments against the new protein target with HDOCK.
4. Filter fragments according to docking sites and docking scores.
5. Replace `list1` and `list2` in `config.py`.
6. Recalculate `policy_num` and `max_len` — required only if the fragment-selection scheme changes; see the note below.
7. Adjust the network dimensions in `cnn_net.py` when necessary — subject to the same condition.
8. Collect new self-play data and retrain the policy-value network. No pre-trained checkpoint is distributed (see [Model weights are not distributed](#model-weights-are-not-distributed)), so every target is trained from a randomly initialized network.

If the fragment-selection scheme described in this work is retained — the five best-docking fragments of each class (single-stranded 3, 4, 5, and 6 nt, plus paired 3, 4, 5, and 6 bp) together with the four single-nucleotide extension actions — the action count stays at 4 + 4 × 5 + 4 × 5 = 44. With `min_len` = 50 and the longest fragment still 6 bp (12 nt), `max_len` remains 49 + 12 = 61, so `policy_num` remains 44 × 50 = 2,200, which is exactly the dimensioning already used by the shipped configuration and by `cnn_net.py`. In that case only the contents of `list1` and `list2` change: steps 6 and 7 can be skipped and no layer in `cnn_net.py` needs to be resized. Recomputing `policy_num` and `max_len` and adjusting the network layers becomes necessary only when the number of fragments retained per class, the fragment length range, or `min_len` changes.

## Reproducibility notes

- Dirichlet noise is added during MCTS action selection, so self-play trajectories are stochastic.
- The trained weights and the replay buffer are not distributed (`current_policy.pkl` and `train_data_buffer.pkl` are created at runtime); rerunning the pipeline regenerates them, but a different random seed yields a different checkpoint (see [Model weights are not distributed](#model-weights-are-not-distributed)).
- Multiple collection processes share the same serialized replay-buffer file without file locking. Concurrent writes can overwrite records or interrupt reads; the multiprocess launcher does not provide synchronized buffer updates.
- For long-running experiments, retain backups of model checkpoints, replay buffers, configuration files, and process logs.
- Computationally generated candidate aptamers still require subsequent experimental validation.

## Citation

If you use AiDTA in your research, please cite:

```bibtex
@article{guo2025aidta,
  title  = {Deep reinforcement learning method for de novo design of nucleic acid aptamers targeting proteins},
  author = {Guo, Gaoxing and Guo, Liangwei and Qian, Jiaqiang and He, Xiaoming and Qian, Xinzhou and Wang, Lei and Huang, Qiang},
  year   = {2025},
  note   = {Unpublished manuscript}
}
```
