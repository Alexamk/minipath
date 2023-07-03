# minipath
Minipath (minimum indirect path) is a graph analysis tool to remove spurious crosslinks and split fused nodes from paired adjacency data.

Specifically implemented for bipartite, weighted graphs of DNA microscopy experimental data.

## Introduction

DNA microscopy is a technique where the location of specific DNA and RNA strands are identified by identifying pairs of adjacent molecules. These are individually recognized by a DNA barcode, called a UMI (Unique Molecule Index).

When errors are introduced in the paired adjacency data, the resulting topological defects can disrupt the proper identification of locations.

Two types of errors are considered here: spurious crosslinks, i.e., connections between random types of nodes independent of location, and fused nodes, i.e., the misidentification of two molecules as the same molecule due to having the same barcode. These errors are identified by the two scripts `edge_filter.py` and `split_fused_nodes.py`. 

## Installation

Python scripts can be run directly in any environment which has the following packages installed (tested version in parentheses).

1. numpy (1.24.3)
2. scipy (1.10.1)
3. numba (0.57.0)
4. scikit-learn (1.2.2)
5. matplotlib (3.7.1)
6. configargparse (1.4)

A .yml file is provided that creates an environment with these packages and python 3.8

`conda env create -f minipath.yml`

## Running the scripts

The scripts can be called as follows 

```
usage: edge_filter.py [-h] -i INFILE -o OUTFOLDER [--label1 LABEL1] [--label2 LABEL2] [--quantiles QUANTILES [QUANTILES ...]]
                      [--cutoffs CUTOFFS [CUTOFFS ...]] [--mode {sparse,dense}]

usage: split_fused_nodes.py [-h] -i INFILE -o OUTFOLDER [--label1 LABEL1] [--label2 LABEL2]
                            [--quantiles QUANTILES [QUANTILES ...]] [--cutoffs CUTOFFS [CUTOFFS ...]]

```

### Example runs 

```
python -i edge_filter.py -i example_input/spurious_crosslinks/concatemers.txt -o sp_test --quantiles 0.01 0.02 0.05 0.1 0.2
```

```
python -i split_fused_nodes.py -i example_input/fused_nodes/concatemers.txt -o fn_test --quantiles 0.01 0.02 0.05
```

### Argument explanation

Input files should be formatted as tab-separated files, with entries

```
label1_nodeid1 label2_nodeid1 edgeweight
label1_nodeid1 label2_nodeid2 edgeweight
...
...
label1_nodeidx label2_nodeidy edgeweight
```
In the example data, label1 is given as `upi_rg` and label2 as `upi_by`. UPI is a shorthand for Unique Polony Index, while `rg` and `by` are arbitrary tags used to differentiate between the two types.

`quantiles` and `cutoffs` are used to determine which edges to remove (`edge_filter.py`) or which nodes to split (`split_fused_nodes.py`). 

For `edge_filter.py`, the cutoff refers to the indirect path values at three steps.

For `split_fused_nodes.py`, the cutoff refers to the normalized cut.

Quantile cutoffs set the cutoff as the lower quantile of all the found relevant data.



