# GNN Super-Resolution Benchmark

## Overview

Welcome to the **GNN Super-Resolution Benchmark** repository. This project focuses on advancing the field of brain graph prediction using Graph Neural Networks (GNNs), specifically in transforming low-resolution (LR) brain graph data into high-resolution (HR) representations efficiently.

This repository contains the code and resources from a Kaggle competition where participants developed and refined machine learning models using the SLIM functional MRI dataset. The competition aimed to evaluate the potential of GNNs in brain graph prediction by leveraging diverse preprocessing, dimensionality reduction, and learning strategies.

## Repository Structure

- `Team1/` to `Team31/` - Folders containing code and resources submitted by each team during the competition.

Each team's folder includes:

- Source code
- Instructions for running their model
- Any additional resources or documentation

## Getting Started

### Prerequisites

To run the code in this repository, you will need:

- Python 3.7 or higher
- PyTorch
- NetworkX
- NumPy
- Additional libraries as specified by each team

### Installation

Clone the repository:

```bash
git clone https://github.com/basiralab/GNN-superresolution-benchmark.git
cd GNN-superresolution-benchmark
```

**Note:** Each team may have different dependencies.

### Dataset

The models are trained and evaluated using the SLIM functional MRI dataset. Due to licensing restrictions, the dataset is not included in this repository.

To obtain the dataset:

1. Visit the SLIM dataset webpage: [SLIM Dataset on NITRC](https://www.nitrc.org/projects/slim/)
2. Follow the instructions to request access.
3. Once downloaded, place the dataset in the appropriate directory as specified by each team's code.


## Evaluation Framework

The models were evaluated using a robust framework that includes:

- Multiple datasets to assess generalization.
- Cross-validation techniques (Random-CV and Cluster-CV).
- Metrics such as Mean Absolute Error (MAE) and Jensen-Shannon Divergence (JSD).

For detailed evaluation metrics and results, refer to the paper associated with this repository.

## Acknowledgments

We extend our gratitude to all the participants of the Kaggle competition for their innovative contributions. This collaborative effort has significantly advanced the research in brain graph super-resolution using GNNs.

---

**Disclaimer:** This repository is for research purposes. Ensure compliance with the SLIM dataset's licensing terms when using the data.