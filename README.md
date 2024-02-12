<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
</p>
<p align="center">
    <h1 align="center">GPTLabs</h1>
</p>
<p align="center">
    <em>Building and Training Generative Pre-trained Transformers Step-by-Step</em>
</p>
<p align="center">
    <em>Developed with the software and tools listed below.</em>
</p>
<p align="center">
    <img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat&logo=GNU-Bash&logoColor=white" alt="GNU Bash">
    <img src="https://img.shields.io/badge/Poetry-60A5FA.svg?style=flat&logo=Poetry&logoColor=white" alt="Poetry">
    <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/ONNX-005CED.svg?style=flat&logo=ONNX&logoColor=white" alt="ONNX">
</p>
<hr>

## Overview

GPTLabs provides a step-by-step implementation guide to developing Generative Pre-trained Transformer (GPT) models, starting from simple embedding layers and advancing to the full transformer architecture. This approach is inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) and enables users to explore the evolution of GPT models. Check out `gptlabs/main.py` for detailed model definitions and commentary on each version's contributions.

### Models Overview

- **Embedding**: Introduces the basic structure with a simple embedding layer.
- **Linear**: Adds a linear transformation layer to learn complex mappings.
- **Positional**: Incorporates positional embeddings for sequence understanding.
- **Attention**: Begins contextual understanding with single-head self-attention.
- **Multi-Attention**: Expands to multi-head attention for simultaneous sequence part attention.
- **Computation**: Adds a FeedForward computation block for enhanced processing.
- **Transformer**: Implements multiple Transformer blocks, each with MultiHeadAttention and FeedForward, including residual connections and layer normalization.
- **GPT**: Enhances the transformer architecture, mirroring the decoder component of GPT-3.
- 
GPTLabs is an invaluable toolkit for those interested in natural language processing, offering efficient data handling, model training, and text generation capabilities.


---


## Getting Started

### Requirements

Before starting, make sure the following dependencies are installed on your system:

- **Python 3.11**: Ensure you have Python version 3.11 installed.
- **Poetry**: Poetry is required for dependency management.

### Installation Steps

Follow these steps to install GPTLabs:

1. **Clone the GPTLabs Repository**:
   Use Git to clone the repository to your local machine.
   ```sh
   git clone https://github.com/pierg/GPTLabs.git
   ```

2. **Navigate to the Project Directory**:
   Change your current directory to the GPTLabs project folder.
   ```sh
   cd GPTLabs
   ```

3. **Install Dependencies**:
   Run the following command to install the project dependencies using Poetry.
   ```sh
   poetry install
   ```

### Running GPTLabs

To start exploring GPTLabs and its model evolution, execute these commands:

1. **Navigate to the GPTLabs Directory**:
   Make sure you are in the GPTLabs directory within the project.
   ```sh
   cd gptlabs
   ```

2. **Execute the Main Program**:
   Use Python to run the main script and begin your exploration.
   ```sh
   python main.py
   ```

---


##  Repository Structure

```sh
└── GPTLabs/
    ├── cleanup.sh
    ├── gptlabs
    │   ├── data
    │   │   ├── __init__.py
    │   │   ├── base_data.py
    │   │   └── text_data.py
    │   ├── main.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── decoder_v1.py
    │   │   ├── decoder_v2.py
    │   │   ├── decoder_v3.py
    │   │   ├── decoder_v4.py
    │   │   ├── decoder_v5.py
    │   │   ├── decoder_v6.py
    │   │   ├── decoder_v7.py
    │   │   └── modules
    │   │       ├── computation.py
    │   │       ├── head.py
    │   │       ├── multi_head.py
    │   │       └── residual.py
    │   ├── optimizers
    │   │   ├── __init__.py
    │   │   ├── adamw.py
    │   │   └── base.py
    │   ├── trainers
    │   │   ├── __init__.py
    │   │   └── trainer.py
    │   └── utils
    │       ├── __init__.py
    │       ├── data.py
    │       ├── math.py
    │       ├── tensors.py
    │       ├── torch.py
    │       └── train.py
    ├── input
    │   └── tiny-shakespeare.txt
    ├── output
    │   └── models
    │       ├── onnx
    │       │   ├── GPT_v1_embedding.onnx
    │       │   ├── GPT_v2_linear.onnx
    │       │   ├── GPT_v3_positional.onnx
    │       │   ├── GPT_v4_attention.onnx
    │       │   ├── GPT_v5_multi-attention.onnx
    │       │   ├── GPT_v6_computation.onnx
    │       │   ├── GPT_v7_gpt.onnx
    │       │   └── GPT_v7_transformer.onnx
    │       ├── torchinfo
    │       │   ├── GPT_v1_embedding.txt
    │       │   ├── GPT_v2_linear.txt
    │       │   ├── GPT_v3_positional.txt
    │       │   ├── GPT_v4_attention.txt
    │       │   ├── GPT_v5_multi-attention.txt
    │       │   ├── GPT_v6_computation.txt
    │       │   ├── GPT_v7_gpt.txt
    │       │   └── GPT_v7_transformer.txt
    │       ├── torchview
    │       │   ├── GPT_v1_embedding.pdf
    │       │   ├── GPT_v2_linear.pdf
    │       │   ├── GPT_v3_positional.pdf
    │       │   ├── GPT_v4_attention.pdf
    │       │   ├── GPT_v5_multi-attention.pdf
    │       │   ├── GPT_v6_computation.pdf
    │       │   ├── GPT_v7_gpt.pdf
    │       │   └── GPT_v7_transformer.pdf
    │       └── torchviz
    │           ├── GPT_v1_embedding.pdf
    │           ├── GPT_v2_linear.pdf
    │           ├── GPT_v3_positional.pdf
    │           ├── GPT_v4_attention.pdf
    │           ├── GPT_v5_multi-attention.pdf
    │           ├── GPT_v6_computation.pdf
    │           ├── GPT_v7_gpt.pdf
    │           └── GPT_v7_transformer.pdf
    ├── poetry.lock
    └── pyproject.toml
```

---

##  Modules

<details closed><summary>.</summary>

| File                                                                              | Summary                                                                                                                                                                                                                                                                                                                                                                                               |
| ---                                                                               | ---                                                                                                                                                                                                                                                                                                                                                                                                   |
| [pyproject.toml](https://github.com/pierg/GPTLabs.git/blob/master/pyproject.toml) | This code snippet is part of the gptlabs repository. It includes various modules such as data handling, models, optimizers, trainers, and utilities. Its main purpose is to provide functionality for training and optimizing GPT (Generative Pre-trained Transformer) models.                                                                                                                        |
| [poetry.lock](https://github.com/pierg/GPTLabs.git/blob/master/poetry.lock)       | The code snippet in this repository is responsible for cleaning up and managing data in the GPTLabs project. It contains a script named `cleanup.sh` and a Python module `base_data.py` that handles base data operations.                                                                                                                                                                            |
| [cleanup.sh](https://github.com/pierg/GPTLabs.git/blob/master/cleanup.sh)         | The code snippet is a cleanup script (`cleanup.sh`) that deletes unnecessary files and directories in the Python project. It removes `__pycache__` directories, `.pyc` files, and the `poetry.lock` file. It also includes deleting the `.venv` directory if present. The script is executed from the root directory of the project and improves the cleanliness and maintainability of the codebase. |

</details>

<details closed><summary>input</summary>

| File                                                                                                | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                 | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [tiny-shakespeare.txt](https://github.com/pierg/GPTLabs.git/blob/master/input/tiny-shakespeare.txt) | This code snippet is part of the GPTLabs repository. It focuses on the data, models, and optimizers modules, providing essential functionality for handling text data, model decoding, and optimization. It plays a critical role in the repository's architecture by enabling efficient data processing and modeling capabilities. Supplementary materials related to this codebase can be found in the GPTLabs repository, including files like `cleanup.sh`, `main.py`, and additional Python modules for data processing, model decoding, and optimization. |

</details>

<details closed><summary>output.models.torchinfo</summary>

| File                                                                                                                              | Summary                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                               | ---                                                                                                                                                                                                                                                                                                                                                                                                             |
| [GPT_v7_gpt.txt](https://github.com/pierg/GPTLabs.git/blob/master/output/models/torchinfo/GPT_v7_gpt.txt)                         | The `GPT_v7_gpt` code snippet is a component of the GPTLabs repository. It's responsible for implementing the GPT (Generative Pre-trained Transformer) model architecture. The code snippet contains multiple layers and implements embedding, multi-head attention, layer normalization, and feed-forward operations. It has 10,788,929 trainable parameters and achieves an estimated total size of 43.37 MB. |
| [GPT_v5_multi-attention.txt](https://github.com/pierg/GPTLabs.git/blob/master/output/models/torchinfo/GPT_v5_multi-attention.txt) | This code snippet represents the GPT_v5 multi-attention model in the GPTLabs repository. It provides an embedding layer, multiple attention heads, and linear layers. The model has a total of 8,609 trainable parameters and is used for text data processing.                                                                                                                                                 |
| [GPT_v6_computation.txt](https://github.com/pierg/GPTLabs.git/blob/master/output/models/torchinfo/GPT_v6_computation.txt)         | This code snippet represents the computation module (GPT_v6) of the GPTLabs repository. It includes embedding, multi-head attention, feedforward, and linear layers. It has a total of 16,961 trainable parameters and outputs a tensor of shape [1, 1, 65].                                                                                                                                                    |
| [GPT_v1_embedding.txt](https://github.com/pierg/GPTLabs.git/blob/master/output/models/torchinfo/GPT_v1_embedding.txt)             | The code snippet in `GPT_v1_embedding.txt` provides information about the GPT_v1 model architecture. It shows the output shape and number of trainable parameters of the Embedding layer in GPT_v1.                                                                                                                                                                                                             |
| [GPT_v4_attention.txt](https://github.com/pierg/GPTLabs.git/blob/master/output/models/torchinfo/GPT_v4_attention.txt)             | The code snippet represents the GPT_v4 model's attention module in the parent repository's architecture. It contains an embedding layer, a head with linear and dropout layers, and a final linear layer. The module has a total of 7,553 trainable parameters.                                                                                                                                                 |
| [GPT_v2_linear.txt](https://github.com/pierg/GPTLabs.git/blob/master/output/models/torchinfo/GPT_v2_linear.txt)                   | The code in the `GPT_v2_linear.txt` file is responsible for defining the architecture of the GPT_v2 model. It consists of an Embedding layer and a Linear layer. The model has a total of 4,225 trainable parameters.                                                                                                                                                                                           |
| [GPT_v7_transformer.txt](https://github.com/pierg/GPTLabs.git/blob/master/output/models/torchinfo/GPT_v7_transformer.txt)         | The code snippet represents the GPT_v7 transformer model in the GPTLabs repository. It consists of stacked blocks of multi-head attention and feedforward layers, resulting in an output shape of [1, 1, 65]. The model has a total of 42,369 trainable parameters.                                                                                                                                             |
| [GPT_v3_positional.txt](https://github.com/pierg/GPTLabs.git/blob/master/output/models/torchinfo/GPT_v3_positional.txt)           | The code snippet in GPT_v3_positional.txt is a part of the GPTLabs repository. It includes the architecture and parameters of the GPT_v3 model, which employs embedding and linear layers to generate output. The model has a total of 4,481 trainable parameters, and its estimated size is 0.02 MB.                                                                                                           |

</details>

<details closed><summary>gptlabs</summary>

| File                                                                        | Summary                                                                                                                                                                                                                   |
| ---                                                                         | ---                                                                                                                                                                                                                       |
| [main.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/main.py) | The code snippet in gptlabs/main.py defines various models with increasing complexity and trains them using specified hyperparameters. It generates text before and after training and saves the model architecture info. |

</details>

<details closed><summary>gptlabs.utils</summary>

| File                                                                                    | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---                                                                                     | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [tensors.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/utils/tensors.py) | This code snippet contains utility functions that provide information about PyTorch tensors. The `print_batch_info` function prints the shapes of input and target tensors, and details of the first batch. The `pretty_print_tensor` function pretty prints the shape, datatype, and the first few entries of a tensor. The `pretty_print_tensor_info` function prints the shape and datatype of a tensor. These functions are part of the `gptlabs/utils/tensors.py` file in the repository's architecture. |
| [train.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/utils/train.py)     | Code snippet `train.py` in the `gptlabs/utils` directory is responsible for training and generating text using a GPT model. It includes functions `train_model` for training the model and `generate` for generating text. The code uses an AdamW optimizer and a cross-entropy loss function.                                                                                                                                                                                                                |
| [math.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/utils/math.py)       | The `math.py` file in the `utils` directory of the `gptlabs` module contains functions for softmax computation, multinomial sampling, and cross-entropy loss calculation using numpy. These functions are critical for various tasks such as natural language processing, machine learning, and neural networks implemented in the parent repository.                                                                                                                                                         |
| [torch.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/utils/torch.py)     | The code snippet in `gptlabs/utils/torch.py` saves model architecture views, model info, and ONNX export files, while also cleaning up any intermediate files. It exports the model to ONNX format, saves a graphical representation of the model, generates a summary of the model, and saves a torchview graph.                                                                                                                                                                                             |
| [data.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/utils/data.py)       | The `data.py` module in the `gptlabs/utils` directory handles data processing tasks such as splitting the data into training and validation sets, loading and tokenizing data, and initializing a batch generator for text data. These functions are crucial for preparing the data for model training and evaluation in the GPTLabs repository.                                                                                                                                                              |

</details>

<details closed><summary>gptlabs.models</summary>

| File                                                                                           | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---                                                                                            | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [decoder_v4.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/decoder_v4.py) | The code snippet `decoder_v4.py` in the `gptlabs/models` directory contains the implementation of the GPT_v4 model. It integrates self-attention into the GPT architecture, allowing the model to focus on relevant parts of the input sequence. The model includes token and positional embeddings, a self-attention head, and a linear layer for language modeling. It can generate text by considering the most recent part of the sequence that fits within its processing capacity.                                                                                                                                                                            |
| [decoder_v1.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/decoder_v1.py) | The `decoder_v1.py` code snippet is a simplified implementation of a Bigram Language Model using an embedding layer. It represents the first step towards building more complex Transformer-based models in the GPTLabs repository. The code initializes the GPT model with an embedding layer mapping vocabulary indices to dense vectors. It performs a forward pass to transform input indices into dense vectors, and can also generate new tokens based on a given starting sequence by iteratively predicting the next token.                                                                                                                                 |
| [decoder_v5.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/decoder_v5.py) | The code snippet in `decoder_v5.py` implements the GPT_v5 model in the parent repository's architecture. This model enhances the GPT architecture by incorporating a multi-head attention mechanism, allowing the model to attend to different parts of the sequence simultaneously. It also includes token and positional embeddings, along with a linear layer for generating logits corresponding to the vocabulary predictions. The `forward` method performs the forward pass of the model, while the `generate` method generates text based on a given starting sequence of token indices.                                                                    |
| [decoder_v2.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/decoder_v2.py) | The code snippet `decoder_v2.py` in the `gptlabs/models` directory is a part of the GPTLabs repository. It implements an evolved version of the GPT model that introduces a linear layer on top of the embeddings. This architecture allows for a more sophisticated mapping from the token embeddings to the vocabulary space, facilitating the learning of richer representations for sequence prediction and generation. The `GPT_v2` class initializes the model with an embedding layer and a linear layer for the language model head. It also includes methods for performing the forward pass and generating new tokens based on a given starting sequence. |
| [decoder_v6.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/decoder_v6.py) | The `decoder_v6.py` code defines the `GPT_v6` model, which extends the GPT architecture by incorporating a multi-head attention mechanism and a feedforward network. It generates predictions for token sequences and can also generate new tokens.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [decoder_v7.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/decoder_v7.py) | The code snippet in `decoder_v7.py` implements the GPT_v7 model, a variant of the GPT architecture. It extends the GPT model with multiple Transformer blocks, deep self-attention mechanisms, and position-wise feedforward networks. The model learns representations of complex sequences and enables effective sequence generation and analysis. It takes token indices as input and produces logits over the vocabulary for each position in the sequence. It also provides a method to generate text by using the most recent part of the sequence and generating new tokens based on the model's processing capacity.                                        |
| [decoder_v3.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/decoder_v3.py) | The `decoder_v3.py` code snippet is a part of the GPTLabs repository's architecture. It provides the GPT_v3 model, which generates text using token and positional embeddings, considering only the most recent tokens within a maximum sequence length. It includes methods for the forward pass to produce logits and for generating new text based on a start sequence.                                                                                                                                                                                                                                                                                          |

</details>

<details closed><summary>gptlabs.models.modules</summary>

| File                                                                                                     | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---                                                                                                      | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [residual.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/modules/residual.py)       | The `residual.py` code snippet defines a Transformer block within a larger codebase. This block contains a multi-head self-attention mechanism followed by a position-wise feedforward network. It applies layer normalization, residual connections, and dropout for regularization. The block's purpose is to process input tensors, stabilize them using normalization, and integrate the original input with the transformed output. Overall, it contributes to the architecture's ability to model complex relationships and perform effective information flow. |
| [multi_head.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/modules/multi_head.py)   | This code snippet is part of the GPTLabs repository's architecture. It implements the Multi-Head Attention module, which runs multiple attention mechanisms (heads) in parallel and combines their outputs. It ensures that the output dimensionality matches the input embedding dimensionality, allowing seamless integration with subsequent layers in the model.                                                                                                                                                                                                  |
| [head.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/modules/head.py)               | The `Head` class in `gptlabs/models/modules/head.py` implements a single head of self-attention mechanism, a key component of Transformer architectures. It calculates attention scores between input embeddings and applies masking and scaling operations to compute the weighted sum of value vectors. The output represents the attention of different input parts and is used for predicting an output.                                                                                                                                                          |
| [computation.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/models/modules/computation.py) | The code snippet in `computation.py` defines the `FeedForward` module used in the Transformer architecture. It contains a feedforward neural network with expansion and compression layers that increase capacity, introduce non-linearity, reduce dimensionality, and apply dropout. This module is responsible for processing input tensors and producing output tensors.                                                                                                                                                                                           |

</details>

<details closed><summary>gptlabs.optimizers</summary>

| File                                                                                     | Summary                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---                                                                                      | ---                                                                                                                                                                                                                                                                                                                                                                                                      |
| [adamw.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/optimizers/adamw.py) | The code snippet `adamw.py` is part of the `gptlabs` package in the repository. It implements the AdamW optimizer, which is a variant of the Adam optimizer with weight decay. This optimizer is used for updating the parameters during the optimization process in the parent repository's architecture. It includes methods for initializing the optimizer and performing a single optimization step. |
| [base.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/optimizers/base.py)   | The code snippet in `base.py` is a base class for all optimizers in the GPTLabs repository. It defines common functionality such as initializing the optimizer, performing optimization steps, and zeroing gradients. Specific optimizers should inherit from this base class and implement the `step` method.                                                                                           |

</details>

<details closed><summary>gptlabs.trainers</summary>

| File                                                                                       | Summary                                                                                                                                                                                                                                                                                                                                              |
| ---                                                                                        | ---                                                                                                                                                                                                                                                                                                                                                  |
| [trainer.py](https://github.com/pierg/GPTLabs.git/blob/master/gptlabs/trainers/trainer.py) | This code snippet represents the Trainer class in the GPTLabs repository. It handles the training loop, loss calculation, and evaluation of a SimpleModule model using a specified optimizer. The class trains the model for a specified number of iterations and batch size, periodically evaluates the model's performance, and prints the losses. |

</details>
