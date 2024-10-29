To implement a version of **VATT** (Video-Audio-Text Transformer) that focuses on **long-form audio and text**, let's break down the **VATT architecture** with the purpose and dimensions of each component. This detailed architecture overview will allow you to understand how VATT handles multimodal inputs, specifically audio and text, and how to adapt it for long-form data.

<img src="https://github.com/user-attachments/assets/f7eab1f5-e020-4ba1-b1f1-a0e636c380d4" width="100%" />

---

### Overview of the VATT Architecture

The VATT model is a **self-supervised multimodal Transformer** that processes **raw input signals** (video, audio waveforms, and text) using **modality-specific or modality-agnostic Transformers**. It was originally designed for tasks like video action recognition and audio event classification. Here’s a step-by-step explanation of its components and architectural flow, focusing on audio and text, with details on each component's purpose and dimensions.

---

### 1. **Input Tokenization**

Each modality (audio and text) is first **tokenized** and **projected** into a common embedding space.

#### Audio Tokenization

- **Input Type**: Raw audio waveform sampled at 48 kHz.
- **Tokenization**: Audio is split into **patches of waveform data**. In the original VATT setup, each audio patch has a length of 128 samples.
- **Linear Projection**: Each patch is passed through a **linear projection layer** that maps it to a 1D token embedding of dimension `d`.
  
  For example, if the input length is 6.4 seconds (153,600 samples at 24 kHz), and patches of 128 samples are used, then there will be `153600 / 128 = 1200` tokens.

#### Text Tokenization

- **Input Type**: Tokenized text (e.g., words or subwords) from audio transcripts.
- **Tokenization**: Text is tokenized using a **one-hot encoding** for each word in the vocabulary, capped to a max sequence length (e.g., 512 tokens).
- **Embedding Layer**: A linear projection converts each one-hot word vector into a token embedding of dimension `d`.
  
  This results in a sequence of text embeddings, where each embedding has dimension `d`.

### 2. **Positional Encoding**

Since transformers are position-agnostic, **positional encodings** are added to each token embedding to encode order information.

- **Audio Positional Encoding**: Audio tokens are given a 1D positional encoding to reflect the temporal order of waveform patches.
- **Text Positional Encoding**: Text tokens are also given a 1D positional encoding based on word order in the sentence.

### 3. **Transformer Encoder**

VATT uses **transformer encoders** with **multi-head self-attention (MHSA)** and **feedforward layers** to process tokens from each modality. In the **modality-specific** version, there’s one encoder per modality (audio, text), while in the **modality-agnostic** version, a single transformer encoder is shared.

Each transformer encoder follows this structure:

#### Multi-Head Self-Attention (MHSA)
- **Purpose**: MHSA allows each token to attend to other tokens in the input sequence, which captures dependencies and correlations between different parts of the data.
- **Dimension of Query, Key, Value (QKV)**: For each head, Q, K, and V vectors are created by linearly transforming the input embeddings. The dimension of each head's Q, K, and V vectors is `d/h`, where `h` is the number of heads.
- **Output Dimension**: After attention is computed for each head, the outputs are concatenated and linearly transformed back to the embedding dimension `d`.

#### Feedforward Layer (MLP)
- **Purpose**: A two-layer feedforward network (typically with a ReLU or GELU activation between layers) is applied to each token's representation, allowing for non-linearity and complexity.
- **Dimensions**: The feedforward layer expands the embedding dimension temporarily (e.g., `4*d`) before reducing it back to `d`.
  
Both MHSA and feedforward layers are followed by **layer normalization** and **residual connections**.

### 4. **Multimodal Projection Head**

After processing through the transformer encoder, each modality’s tokens are aggregated into a single representation (often through a designated [CLS] token or an average pooling operation). This representation is then passed through a **projection head**.

#### Multimodal Projection Head

- **Purpose**: Projects each modality’s representation into a **common representation space**, allowing the model to align the embeddings from different modalities.
- **Hierarchical Common Space**: VATT defines a semantically hierarchical common space where audio, video, and text representations are projected based on their level of semantic granularity.
  
  For audio-text only, you might skip the video projection space and align audio and text representations in a shared **audio-text space**.

### 5. **Contrastive Learning Objectives**

To learn from **unlabeled data**, VATT uses **contrastive learning** to encourage representations from corresponding pairs (e.g., audio-text pairs from the same instance) to be close in the common space, while non-corresponding pairs are pushed apart.

#### Noise Contrastive Estimation (NCE)
- **Purpose**: Maximizes similarity between positive pairs (e.g., audio-text pairs from the same instance) while minimizing it for negative pairs (e.g., mismatched pairs).
- **Process**: Calculates similarity scores between positive and negative pairs and applies a softmax over these scores with a temperature parameter `τ`.

### 6. **DropToken (for Efficient Training)**

To handle long sequences, VATT introduces **DropToken** during training, which randomly drops a portion of input tokens to reduce computational load.

- **Purpose**: Reduces training time and memory usage by processing fewer tokens, making it feasible to train on long sequences.
- **Effect on Long Data**: For your setup with long-form audio and text, DropToken will help manage the high computational costs associated with long sequences.

---

### Dimensions and Configurations in VATT

Here’s a summary of the typical dimensions and configurations for each component in VATT:

| Component                       | Input Dimension               | Output Dimension            |
|---------------------------------|-------------------------------|-----------------------------|
| **Audio Tokenization**          | `(N, L_a)` (audio waveform)   | `(N, T_a, d)`               |
| **Text Tokenization**           | `(N, L_t)` (text tokens)      | `(N, T_t, d)`               |
| **Positional Encoding**         | `(N, T, d)`                   | `(N, T, d)`                 |
| **Transformer Encoder**         | `(N, T, d)`                   | `(N, T, d)`                 |
| **Q, K, V in MHSA**             | `(N, T, d)`                   | `(N, T, d/h)` per head      |
| **Feedforward Layer (MLP)**     | `(N, T, d)`                   | `(N, T, d)`                 |
| **Multimodal Projection Head**  | `(N, d)`                      | `(N, d_proj)`               |

- **N**: Batch size
- **L_a**: Length of audio input (total samples)
- **L_t**: Length of text input (number of tokens)
- **T_a, T_t**: Number of tokens after tokenization for audio and text, respectively
- **d**: Embedding dimension (e.g., 512 or 768)
- **h**: Number of attention heads
- **d_proj**: Projected dimension in the common space (e.g., 256 or 512)

---

### Modifications for Long-Form Data

To adapt VATT for **long-form audio and text**, we will consider the following changes:

1. **Increase Token Length**: Tokenize long-form audio into more segments and longer text sequences. Set a maximum sequence length for practicality, potentially using DropToken to reduce computational load.

2. **Adjust Positional Encodings**: Ensure that positional encodings are sufficiently large to cover the increased sequence lengths in audio and text.

3. **Expand Transformer Layers**: Consider adding more transformer layers to manage the complexity of longer dependencies in the data.

4. **Experiment with DropToken Rates**: DropToken will help control the computation. Experiment with higher drop rates for long-form data to balance efficiency and performance.

---

### Summary

- **VATT** uses **transformer encoders** to process modality-specific tokenized inputs and align them in a **common representation space**.
- **Audio and Text Processing**: Tokenized into smaller segments, projected, and encoded through positional embeddings.
- **Contrastive Learning**: The model learns to align multimodal embeddings through **contrastive learning objectives** in the shared representation space.
- **Adaptations for Long-Form Data**: Increase sequence length, utilize DropToken, and consider more transformer layers to capture long-range dependencies.

Implementing these changes will allow us to explore how VATT performs with **long-form audio and text** while keeping the structure and training methodology as close as possible to the original model.
