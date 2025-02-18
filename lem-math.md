# Mathematical Formulation of Large Event Model (LEM)

## 1. Problem Definition

Let us define the core problem formally. Given:
- An interruption event $I$ at time $t$
- Knowledge Graph $G = (V,E)$ representing the smart home context
- Historical event sequence $\mathcal{H}$

Our objective is to predict optimal appliance settings $S^*$:

$S^* = \argmax_S P(S | I, t, G, \mathcal{H})$

## 2. Knowledge Graph Neural Network

### 2.1 Graph Structure

The Knowledge Graph $G = (V,E)$ consists of:
- Vertex set $V$ containing different types of nodes (appliances, users, activities)
- Edge set $E$ representing relationships between nodes
- Edge types $\mathcal{R}$ (e.g., "uses", "participates_in", "located_in")

### 2.2 R-GCN Layer

For each node $i$, the Relational Graph Convolutional Network (R-GCN) update is:

$h_i^{(l+1)} = \sigma\left(\sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} + W_0^{(l)} h_i^{(l)}\right)$

where:
- $h_i^{(l)}$ is the hidden state of node $i$ at layer $l$
- $\mathcal{N}_i^r$ is the set of neighbors under relation $r$
- $c_{i,r}$ is a normalization constant
- $W_r^{(l)}$ are relation-specific weight matrices
- $\sigma$ is a non-linear activation function

## 3. Transformer Architecture

### 3.1 Input Representation

The input sequence $S$ is constructed as:

$S = [e_I, e_t, e_{G_1}, e_{G_2}, ..., e_{G_n}]$

where:
- $e_I$ is the interruption embedding
- $e_t$ is the time embedding
- $e_{G_i}$ are Knowledge Graph node embeddings

### 3.2 Positional Encoding

For position $pos$ and dimension $i$:

$PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$
$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d})$

### 3.3 Multi-Head Attention

The attention mechanism is defined as:

$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

For multi-head attention:

$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$

where each head is:

$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

## 4. Output Layer

The final prediction layer computes appliance settings probabilities:

$\hat{S} = \sigma(\text{Transformer}([e_I, e_t, e_G])W_S + b_S)$

where:
- $\sigma$ is the sigmoid activation function
- $W_S, b_S$ are the output layer weights and biases

## 5. Loss Function

The model is trained using binary cross-entropy loss:

$\mathcal{L} = -\sum_i (S_i\log\hat{S}_i + (1-S_i)\log(1-\hat{S}_i))$

where:
- $S_i$ is the true setting for appliance $i$
- $\hat{S}_i$ is the predicted probability for appliance $i$

## 6. Time Representation

Time is encoded using cyclical features:

$\text{Hour}_{\sin} = \sin\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$
$\text{Hour}_{\cos} = \cos\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$

This creates a continuous, cyclical representation of time that captures daily patterns.

## 7. Training Procedure

The model is trained end-to-end using:
1. Joint optimization of GNN and Transformer parameters
2. Mini-batch gradient descent with Adam optimizer
3. Teacher forcing during training for sequence prediction
4. Gradient clipping to prevent exploding gradients

The complete parameter set $\theta$ includes:
- R-GCN weights $\{W_r^{(l)}\}$ for all relations and layers
- Transformer attention weights and biases
- Output layer parameters $W_S, b_S$
- Embedding matrices for interruptions and time features

## 8. Complexity Analysis

For a graph with $|V|$ nodes and $|E|$ edges:
- GNN computation: $O(|E| \cdot d)$ where $d$ is embedding dimension
- Transformer attention: $O(n^2 \cdot d)$ where $n$ is sequence length
- Total per-batch complexity: $O(|E| \cdot d + n^2 \cdot d)$

This formulation provides a complete mathematical framework for implementing and training the Large Event Model.
