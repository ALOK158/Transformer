# Transformer Language Model (nanoGPT-style)

## Project Description

This project implements a GPT-style transformer language model from scratch in PyTorch, inspired by Andrej Karpathy’s nanoGPT and the original Transformer architecture. The model is designed for autoregressive text generation and can be trained on any character- or token-level dataset. The implementation closely follows modern best practices in transformer design, including multi-head self-attention, residual connections, pre-layer normalization, and learned positional embeddings.

Trained over text -tinyShakespeare data.

## Model Architecture

- **Embedding Layer:**  
  - Token embeddings: Maps input indices to dense vectors.
  - Positional embeddings: Learns position information for each token in the sequence.

- **Stack of Transformer Blocks:**  
  - Each block contains:
    - Multi-head self-attention (with causal masking)
    - Feedforward network (MLP with GELU activation)
    - Residual connections
    - Layer normalization (pre-norm)

- **Final LayerNorm:**  
  - Applied after the last transformer block for output stabilization.

- **Output Head:**  
  - Linear layer projecting to vocabulary logits for next-token prediction.

## Key Hyperparameters

| Parameter         | Value      | Description                                             |
|-------------------|------------|--------------------------------------------------------|
| batch_size        | 64         | Number of sequences processed in parallel              |
| block_size        | 256        | Maximum context length (sequence length)               |
| n_embd            | 384        | Embedding and model hidden size                        |
| n_head            | 6          | Number of attention heads per transformer block        |
| n_layer           | 6          | Number of stacked transformer blocks                   |
| dropout           | 0.2        | Dropout rate for regularization                        |
| learning_rate     | 3e-4       | AdamW optimizer learning rate                          |
| max_iters         | 5000       | Total training iterations                              |
| eval_interval     | 500        | Interval for evaluation and loss printing              |
| eval_iters        | 200        | Batches used for evaluation at each interval           |

## References and Papers Consulted

| Paper / Resource                                                                 | Contribution/Usage in Project              |
|----------------------------------------------------------------------------------|--------------------------------------------|
| [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)                               | Core transformer architecture              |
| [Language Models are Few-Shot Learners (Brown et al., 2020, GPT-3)](https://arxiv.org/abs/2005.14165)              | GPT-style autoregressive generation        |
| [GPT-2: Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | LayerNorm placement, causal masking        |
| [nanoGPT (Andrej Karpathy, 2023)](https://github.com/karpathy/nanoGPT)                                                | Minimal, educational GPT implementation    |
| [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)                                                          | Deep learning framework and API reference  |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)                                                          | Residual learning framework to ease the training of networks that are substantially deeper  |
## Features

- Modern transformer block design (pre-norm, residuals)
- Multi-head self-attention with causal mask
- Learned positional embeddings
- Configurable depth and width
- Training and validation loss monitoring
- Ready for text generation after training

## Usage

1. **Set up dataset and vocabulary**
2. **Configure hyperparameters as needed**
3. **Train the model with the provided training loop**
4. **Monitor loss and save checkpoints**
5. **Generate text samples using the `generate` method**

## To Do

- [ ] Add training/validation loss curve graph
- [ ] Add sample text generations
- [ ] Still need to work on the increasing  the Accuracy of the model
- [ ] Experiment with different datasets and hyperparameters

## Acknowledgements

This project draws inspiration and guidance from the open-source work of Andrej Karpathy and the foundational transformer literature. See the references above for the primary sources that shaped this implementation.

*For questions or contributions, please open an issue or submit a pull request.*
