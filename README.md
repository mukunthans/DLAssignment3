# Tamil Transliteration with Seq2Seq Models

This repository contains implementations of sequence-to-sequence models for Tamil transliteration - the process of converting English text to Tamil script. Two variants of the model are provided: one with attention mechanism and one without, along with advanced features like beam search decoding and comprehensive visualization tools.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
  - [Basic Seq2Seq Model](#basic-seq2seq-model)
  - [Attention-based Seq2Seq Model](#attention-based-seq2seq-model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results Comparison](#results-comparison)
- [References](#references)

## Overview

Transliteration is the process of converting text from one script to another while preserving the pronunciation. This project focuses on transliterating English text (Latin script) to Tamil text (Tamil script). 

The implementations use PyTorch-based sequence-to-sequence (Seq2Seq) models with different variants:
1. **Basic Seq2Seq Model**: A standard encoder-decoder architecture without attention
2. **Attention-based Seq2Seq Model**: An enhanced architecture with attention mechanism that helps the model focus on relevant parts of the input sequence

Both implementations include comprehensive data processing, training, evaluation, and visualization capabilities, along with Weights & Biases (wandb) integration for experiment tracking and hyperparameter optimization.

## Features

### Common Features in Both Implementations
- Character-level encoding for handling both English and Tamil scripts
- Support for multiple RNN cell types (RNN, LSTM, GRU)
- Teacher forcing during training
- Greedy decoding during inference
- WandB integration for experiment tracking
- Comprehensive evaluation metrics
- CSV export of predictions
- HTML visualization of results

### Features Unique to Attention-based Model
- Additive attention mechanism
- Beam search decoding
- Attention weight visualization
- Comparative analysis between greedy and beam search
- Enhanced hyperparameter tuning capabilities
- Heat map visualizations of attention patterns

## Requirements

```
torch>=1.7.0
numpy>=1.19.0
pandas>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
wandb>=0.10.0
IPython>=7.0.0
```

## Dataset

The implementations use the Aksharantar dataset for Tamil transliteration. The dataset is organized into three splits:

- `tam_train.csv`: Training data
- `tam_valid.csv`: Validation data
- `tam_test.csv`: Test data

Each CSV file contains pairs of English text and corresponding Tamil transliterations without headers.

Data paths can be configured in the code:

```python
DATA_PATHS = {
    "train": "/path/to/tam_train.csv",
    "test": "/path/to/tam_test.csv",
    "valid": "/path/to/tam_valid.csv"
}
```

## Model Architecture

### Basic Seq2Seq Model

The basic model consists of an encoder and a decoder without attention:

1. **Encoder**: 
   - Character embedding layer
   - Multi-layer RNN (configurable cell type: RNN, LSTM, or GRU)
   - Dropout for regularization

2. **Decoder**:
   - Character embedding layer
   - Multi-layer RNN
   - Linear output layer
   - Log softmax activation

### Attention-based Seq2Seq Model

The attention model builds upon the basic model by adding an attention mechanism:

1. **Encoder**: 
   - Same as the basic model

2. **Attention Mechanism**:
   - Additive attention (Bahdanau-style)
   - Aligns decoder states with encoder outputs
   - Produces attention weights and context vectors

3. **Decoder with Attention**:
   - Character embedding layer
   - Attention-weighted context
   - Multi-layer RNN with concatenated embedding and context
   - Linear output layer
   - Log softmax activation

## Training

Both implementations support a similar training workflow:

```python
# Default hyperparameters
HYPERPARAMS = {
    "char_embed_dim": 256,
    "hidden_size": 256,
    "batch_size": 256,
    "num_layers": 2,
    "learning_rate": 0.001,
    "epochs": 15,
    "cell_type": "GRU",
    "dropout": 0.2,
    "optimizer": "nadam",
    "teacher_forcing_ratio": 0.5
}

# Run main function to start training
if __name__ == "__main__":
    main()
```

Training progress is logged to WandB, including:
- Training/validation loss
- Training/validation accuracy
- Learning rate changes
- Best model checkpoints

## Evaluation

Both implementations support comprehensive evaluation:

```python
# For basic model
encoder, decoder = train_model(HYPERPARAMS, train_loader, val_loader, data, test_source, test_target)

# For attention model
evaluator = Evaluator(model, processor)
evaluator.generate_predictions_csv(test_source, test_target, use_beam_search=False)
```

The evaluation process includes:
- Character-level accuracy
- Word-level accuracy
- CSV export of predictions
- Test set evaluation
- Error analysis

## Visualization

### Basic Model
The basic model includes HTML-based visualization of predictions:

```python
inputs = [r['input'] for r in results]
preds = [r['prediction'] for r in results]
targets = [r['target'] for r in results]
display_prediction_results(inputs, preds, targets)
```

### Attention Model
The attention model includes additional visualization capabilities:

```python
# Visualize attention patterns
evaluator.visualize_attention(test_source, test_target, num_samples=6)

# Compare greedy vs beam search
evaluator.compare_decoding_methods(test_source, test_target, num_samples=10)
```

Visualizations include:
- Attention heat maps
- Comparison tables
- Attention weight matrices
- Matplotlib figures exportable to various formats

## Hyperparameter Tuning

Both implementations support hyperparameter tuning with WandB Sweep:

```python
# Run hyperparameter sweep
tuner = HyperparameterTuner(data_manager)
tuner.run_sweep(count=20)
```

Tunable parameters include:
- Learning rate
- Embedding dimension
- Hidden size
- Number of layers
- Batch size
- Cell type (RNN, LSTM, GRU)
- Dropout rate
- Optimizer
- Teacher forcing ratio (for attention model)
