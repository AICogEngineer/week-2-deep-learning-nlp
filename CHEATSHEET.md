# Weekly Cheatsheet: Deep-Learning-NLP


## Weekly Overview

This week covered Deep Learning NLP. By the end, trainees can:
- Embedding
- Recurrent Neural Networks
- Encoding
- Transformers

---

## Concept Quick Reference

| Concept | Definition | Key Use Case |
|---------|------------|--------------|
| TensorBoard | Tensorflow tool that shows graphs of model processes over time | Visualizing model loss, accuracy, and performance|
| Autoencoder | Process of denoising and shrinking an image to feature maps | Analyzing features in large images |
| Reconstruction Loss | An image getting blurry / losing some features when it is decoded | Happens when decoding images |
| Latent Space | A lower dimensional representation of high dimensional data | Representing embeddings or other high dimensional graphs |
| Backpropagation | Moving back through a model in order to see how much each weight and bias contributed to results | Analyzing and modifying individual eights and biases for fine tuning |
| Gradient Descent | Used to find the minimum value and minimize error | _____ |
| Learning Rate | How fast a model learns / is adjusted at each step | Very important hyperparameter in every model |
| Batch Normalization | Centers data around a mean of zero to stabilize data | Keeps gradients stable and and speeds up convergence, especially for larger models |
| Tokenization | The process of grouping together characters in LLMs | Basically anytime human language is being analyzed |
| BPE (Byte-Pair Encoding) | Tokenization done by grouping together characters based on how frequently they are adjacent | Smaller vocabulary and can handle unknown words |
| One-Hot Encoding | Creates a true-false vector that includes a value for every word | Very large vocabulary size, very high dimensional space, not really great |
| Word Embedding | Embedding values to show how related each word is to each other word | Reduces dimensionality and gets rid of mostly zeros |
| Word2Vec (Skip-gram/CBOW) | Algorithm that creates vectors from to represent words | Provides efficient word embeddings at the cost of context |
| RNN | Recurrent neural network - feeds output back into a hidden layer | When you need to preserve context |
| LSTM | Long term short memory - neural network that includes gates to select what memory to keep | Allows the model to handle more context and run for longer |
| GRU | Acts as basically a middle ground between simple RNNs and LSTMs | _____ |
| Vanishing Gradient | Problem where gradients become extremely small, essentially negligible, over time | Happens during backpropagation |
| Sequence Masking | _____ | _____ |
| Early Stopping | The act of a model stopping early when loss and validation loss start to diverge | Basically any time you want a good model |
| Dropout | Killing / Deactivating random neurons during training | Reduces over-fitting by reducing the amount of very intense weights |
| L1 Regularization | _____ | _____ |
| L2 Regularization | _____ | _____ |
| Data Augmentation | _____ | _____ |
| Attention Mechanism | _____ | _____ |
| Self-Attention | _____ | _____ |
| Query (Q) | _____ | _____ |
| Key (K) | _____ | _____ |
| Value (V) | _____ | _____ |
| Multi-Head Attention | _____ | _____ |
| Transformer | _____ | _____ |
| Positional Encoding | _____ | _____ |
| Layer Normalization | _____ | _____ |
| Residual Connection | _____ | _____ |
| Feed-Forward Network (FFN) | _____ | _____ |

---

## Pros & Cons

### Optimizer Comparison

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| SGD | More accurate, less divergence | Takes more time | Stability, precision, using momentum |
| Adam | Very fast, quick convergence| Much more likely to overfit | General prototyping, fast learning |
| RMSprop | </li><li> Automatically adjusts learning rates </li><li> Works well with non-stationary objectives </li><li> Handles sparse gradients effectively | _____ | _____ |

### Text Encoding Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| One-Hot Encoding | No unknown words, highly supported | Very high dimensionality, 99.9% zeros | limited data and unordered features |
| Word Embeddings | Retains context between words / meaning | Out of vocabulary problem | Finding semantic similarity and general text analysis |
| Pretrained Embeddings | Reduced cost and resources | Limited use cases / non specific | Using languages that the embedding has been pretrained on |

### Sequential Models

| Architecture | Pros | Cons | Best For |
|--------------|------|------|----------|
| Simple RNN | Fewer parameters, fast to train, simple architecture | Suffers from vanishing gradients, poor long-term memory | Short sequences, simple temporal patterns |
| LSTM | Strong long-term memory via gates, handles vanishing gradients well | More parameters, slower training, higher compute | Long sequences, language modeling, time-series with long dependencies |
| GRU | Fewer parameters than LSTM, faster training, good performance | Slightly less expressive than LSTM | Medium-length sequences, when speed and performance trade-off is needed |



### Regularization Techniques

| Technique | Pros | Cons | Best For |
|-----------|------|------|----------|
| Dropout | Reduces overfitting, easy to apply | Slower learning | Deep neural networks |
| L1 Regularization | Creates sparse weights, feature selection | Harder optimization | Models needing sparsity |
| L2 Regularization | Keeps weights small, stabilizes training | Does not remove features | Most neural networks |
| Early Stopping | Simple implementation, prevents overfitting| Requires validation data | Any model |

### Attention vs RNN

| Aspect | RNN-based | Attention-based | Best For |
|--------|-----------|-----------------|----------|
| Long-range dependencies | Struggles, fades over time  | Direct Connections Between Tokens | Long Sequences, Context Heavy Tasks|
| Training speed | Step By Step (Slower)| Parallelizable (Faster)| Large Datasets, Speed Critical Tasks|
| Interpretability | Hidden States| Weights Interpretable| Explainable NLP|
| Memory efficiency | Moderate| Memory Intensive| Limited Resources = RNN Could Be Better|

### Transformer Architectures

| Architecture | Pros | Cons | Best For |
|--------------|------|------|----------|
| Encoder-only (BERT) | Strong understanding of context, bidirectional | Cannot generate text | Classification, QA |
| Decoder-only (GPT) | Text generation, strong language modeling | Unidirectional context | Text generation, chatbots, code completion |
| Encoder-Decoder (T5) | Flexible input-output mapping | High compute cost | Translation, summarization |

—	

## When to Use What

### Choosing a Text Encoding

| If you have... | And you need... | Then use... | Because... |
|----------------|-----------------|-------------|------------|
| Small vocabulary | Simplicity | __One-Hot___ | _____ |
| Large vocabulary | Semantic meaning | __Embeddings___ | _____ |
| Limited training data | Transfer learning | _Pretrained____ | _____ |

### Choosing a Sequential Model

| If your sequences are... | And you need... | Then use... | Because... |
|--------------------------|-----------------|-------------|------------|
| Short (<10 tokens) | Simple architecture | __RNN___ | _____ |
| Long (50+ tokens) | Long-term memory | __LSTM___ | _____ |
| Variable length | Efficient training | __GRU___ | _____ |

### Handling Overfitting

| If you observe... | Then try... | Because... |
|-------------------|-------------|------------|
| Train acc high, val acc low | Drop Out | Reduces Overfitting (Randomly Disabling Neurons)|
| Large weight values | L2 Regularization | Penalizes Large Weights, Stabilizes Training|
| Validation loss increasing | Early Stopping | Stops Training Before Overfitting Occurs|

### Choosing Attention vs Transformer

| If you need... | And your constraint is... | Then use... | Because... |
|----------------|---------------------------|-------------|------------|
| Sequence-to-sequence with context | Moderate compute | _____ | _____ |
| Best performance on NLP | Sufficient GPU memory | _____ | _____ |
| Very long sequences (>512) | Memory limited | _____ | _____ |
| Real-time inference | Latency critical | _____ | _____ |

---

## Essential Commands

### TensorBoard (Monday)

```python
# Set up TensorBoard callback
tensorboard_callback = tf.keras.callbacks.Tensorboard(log_dir='./logs')

# Launch TensorBoard (in terminal)
# tensorboard --logdir=’logs/subdir’
```

### Autoencoder Architecture (Monday)

```python
# Encoder
encoder = keras.Sequential([
    layers.Dense(node_num, activation='relu', input_shape=(input_vvec_size,)),
    layers.Dense(node_num, activation='relu'),  # Latent space
])

# Decoder
decoder = keras.Sequential([
    layers.Dense(node_num, activation='relu'),
    layers.Dense(node_num, activation='relu'),  # Reconstruct original
])
```

### Backpropagation (Tuesday)

```python
# Manual gradient computation with GradientTape
with tf.GradientTape() as tape:
    predictions = model(x_train)
    loss = loss_fn(y_train, predictions)

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variable))
```

### Batch Normalization (Tuesday)

```python
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),  # Add batch normalization
    layers.Dense(32, activation='relu'),
])
```

### Tokenization (Wednesday)

```python
# Word-level tokenization
tokenizer =TokenizerType()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Access vocabulary
vocab_size = len(tokenizer.word_index)
word_index = tokenizer.word_index
```

### BPE/Subword Tokenization (Wednesday)

```python
# Byte-Pair Encoding for handling OOV words
# Breaks words into subword units
# Example: "unhappiness" → ["un", "happiness"]

# Common libraries: SentencePiece, Hugging Face tokenizers
```

### Word2Vec Models (Wednesday)

```python
# Skip-gram: Predicts context words from target word
# CBOW: Predicts target word from context words

# Word arithmetic
# king - man + woman ≈ queen
# paris - france + italy ≈ rome
```

### Word Embeddings (Wednesday)

```python
# Keras Embedding layer
model.add(layers.Embedding(
    input_dim=vocab_size,      # Vocabulary size
    output_dim=dim,     # Embedding dimension
    input_length=sequence_length    # Sequence length
))
```

### RNN/LSTM (Thursday)

```python
# Simple RNN
model.add(layers.SimpleRNN(units=_____, activation='_____'))

# LSTM (preferred for long sequences)
model.add(layers.LSTM(units=_____, return_sequences=_____))
```

### Sequence Padding (Thursday)

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

```

### Sequence Masking (Thursday)

```python
# Masking tells model to ignore padded values
model.add(layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    mask_zero=_____  # Treat 0 as padding
))

# Or use explicit Masking layer
model.add(layers.Masking(mask_value=_____))

### Saving & Loading Models (Friday)

```python
# Save entire model (architecture + weights)
model.save('model.h5')  # _____ format
model.save('model_dir')  # _____ format (recommended)

# Load model
loaded_model = tf.keras.models._____(_____ )

# Save/load only weights
model.save_weights('weights.h5')
model.load_weights(_____)
```

### Model Checkpoints (Friday)

```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_{epoch:02d}_{_____:.2f}.h5',
    monitor='_____',
    save_best_only=_____,
    mode='_____'  # 'min' for loss, 'max' for accuracy
)

### Early Stopping (Friday)

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='_____',
    patience=_____,
    restore_best_weights=_____
)
```

### Regularization (Friday)

```python
# Dropout layer
model.add(layers.Dropout(_____))

# L2 Regularization in Dense layer
model.add(layers.Dense(64, kernel_regularizer=tf.keras.regularizers._____(_____)))
```

### Attention Mechanism (Friday)

```python
# Scaled Dot-Product Attention formula
# Attention(Q, K, V) = softmax(Q · K^T / sqrt(_____)) · V

# Self-Attention: Q, K, V all from same input
query = layers.Dense(_____)(inputs)
key = layers.Dense(_____)(inputs)
value = layers.Dense(_____)(inputs)

# Compute attention scores
scores = tf.matmul(query, key, transpose_b=_____)
scores = scores / tf.math.sqrt(tf.cast(d_model, tf.float32))
attention_weights = tf.nn._____(scores, axis=-1)
output = tf.matmul(attention_weights, _____)
```

### Multi-Head Attention (Friday)

```python
# Using Keras built-in MultiHeadAttention
attention = layers.MultiHeadAttention(
    num_heads=_____,           # Number of parallel attention heads
    key_dim=_____,             # Dimension per head (d_model / num_heads)
    dropout=_____              # Dropout rate for attention weights
)

# Self-attention: query and value are the same
output = attention(_____, _____)  # (query, value)
```

### Transformer Encoder Block (Friday)

```python
# Encoder block structure:
# Input → MultiHeadAttention → Add & Norm → FFN → Add & Norm → Output

class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='__relu___'),  # Expand (typically 4x d_model)
            layers.Dense(d_model),                       # Project back to d_model
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
    
    def call(self, x):
        attn_output = self.attention(x, x)
        x = self.layernorm1(x + attn_output)  # Residual connection
        ffn_output = self.ffn(x)
        return self.layernorm2(x + ffn_output)  # Residual connection
```

### Positional Encoding (Friday)

```python
# Why needed: Transformers have no built-in _____ information
# Solution: Add position encoding to embeddings

# Learnable positional embeddings
max_seq_len = 100
d_model = 512

# Token embedding + position embedding
token_embed = layers.Embedding(vocab_size, d_model)(inputs)
positions = tf.range(max_seq_len)
pos_embed = layers.Embedding(max_seq_len, d_model)(positions)
final_embed = token_embed + pos_embed
```

---

## Common Gotchas

| Topic | Wrong | Right |
|-------|-------|-------|
| Backprop | Not detaching gradients when needed | _____ |
| Batch Norm | Placing batch norm after activation | _____ |
| Tokenization | Not handling OOV (out-of-vocabulary) words | _____ |
| Embeddings | Using randomly initialized embeddings on small data | _____ |
| RNN | Using RNN for very long sequences | _____ |
| LSTM | Forgetting to set return_sequences for stacked LSTMs | _____ |
| Padding | Padding with zeros without masking | _____ |
| Overfitting | Adding more layers when already overfitting | _____ |
| Early Stopping | Setting patience too low | _____ |
| Saving Models | Only saving weights, not architecture | _____ |
| Attention | Not scaling dot product by √d_k | _____ |
| Multi-Head | Using wrong key_dim (should be d_model/num_heads) | _____ |
| Positional Encoding | Forgetting to add position info to embeddings | _____ |
| Transformer | Not using residual connections | _____ |
| Layer Norm | Using Batch Norm instead of Layer Norm in Transformers | _____ |
| FFN | Not expanding dimension in feed-forward (should be 4x) | _____ |

---

## Key Formulas

### Backpropagation (Chain Rule) - THE CORE FORMULA
```
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w

Where:
- L = A scalar value that measures how wrong the model’s prediction (loss function)
- a = The output of a neuron after applying the activation function (activation output)
- z = The linear combination of inputs and weights before activation(weighted sum before activation)
- w = A trainable parameter controlling the influence of an input (weight being updated)

∂L/∂a
 → How much does the loss change if the neuron’s output changes?

∂a/∂z
 → How sensitive is the activation function?
 (e.g., ReLU, sigmoid, tanh)

∂z/∂w
 → How much does the weighted sum change if a weight changes?

Forward:  Input → z = Σ(w·x) + b → a = activation(z) → Output → Loss
Backward: Loss → ∂L/∂a → ∂a/∂z → ∂z/∂w → ∂L/∂w (weight update)
 w_new = w_old − α × ∂L/∂w
```

### Gradient Descent Update
```
w_new = w_old - α × ∂L/∂w

Where:
- α =Controls how big each update step is: Too large → training diverges, Too small → training is very slow (learning rate)
- ∂L/∂w = Gradient: direction and magnitude of steepest loss increase (gradient of loss w.r.t. weight)

```

### Momentum Update (SGD with Momentum)
```
v_t = β × v_{t-1} + (1 - β) × ∂L/∂w
w_new = w_old - α × v_t

Where:
- β = momentum coefficient (typically 0.9)
- v = velocity (accumulated gradient)
- ∂L/∂w = current gradient of the loss with respect to weights
- α = learning rate

```

### L2 Regularization Loss
```
Total Loss = Original Loss + λ × Σ(w²)

Where:
- λ = hyperparameter that controls how strongly a model is penalized for having large or complex weights during training (regularization strength)
- Σ(w²) = single number that measures how large all the model’s weights are overall (sum of squared weights)
```

### Dropout (Training vs Inference)
```
Training:   output = input × mask / (1 - p)
Inference:  output = input × 1

Where:
- p = % of neurons dropped (dropout probability)
- mask = random binary mask (keep 1s and drop 0s)
```

### Embedding Lookup
```
Vocabulary Size: V = number of unique tokens
Embedding Dim:   D = vector size
Embedding Matrix: V × D

Lookup: word_index → row i of embedding matrix → vector of dim D
```


