# DDGSR source code

This package contains the DDGSR implementation for Mashup-API recommendation.
DDGSR combines dynamic interest/disinterest graphs, asymmetric view fusion,
and a semantic residual modulated by disagreement between the two views.
The main configuration uses BPR training without contrastive loss.

## Setup

Use Python 3.10 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare the following files locally:

```text
data/tssgcf_official/Data/
    train.txt
    test.txt
    mashup_descr_emb.pt
    api_descr_emb.pt
```

Each interaction line contains a zero-based Mashup ID followed by its API IDs.
The current loader expects 2,289 Mashups and 956 APIs. The two text-feature
files contain aligned Mashup and API tensors; the main configuration uses
384-dimensional input text features and 256-dimensional PCA features.
Use the same interaction split, node ordering and text features for every method.
Data and pretrained text features are not distributed in this source package;
they must be obtained separately. Arbitrary re-splitting will not reproduce the
same evaluation protocol.

## Run

From this directory:

```bash
python run_experiment.py
python run_experiment.py --device cpu --seeds 42
python run_experiment.py --dry-run
```

The default training seeds are 42, 100, 2024, 3407 and 8080, defined centrally
in `experiment_config.py`. Python, NumPy and PyTorch use the run seed, including
DDGSR embedding initialization. Validation splitting and initial negative-graph
construction use fixed protocol seeds of 100. These fixed protocol seeds keep
the split and initial graph stable across training runs.
Evaluation cutoffs are 5, 10, 15 and 20. The main runner trains for 22 epochs.
For optional variants and ablations, see `python train.py --help`.

New runtime artifacts are written under local `outputs/`; no experiment results,
logs, checkpoints or data are included in this source distribution.
Exact bitwise reproducibility across different hardware is not guaranteed.

## Tests

```bash
python -m pytest -q
```
