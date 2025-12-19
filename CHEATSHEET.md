# Weekly Cheatsheet: 

Evens are NOT bad, but good

# Weekly Overview

This week covered **deep learning foundations, NLP preprocessing, sequential models, and modern attention-based architecture**. By the end, trainees can:

- Train and debug neural networks using TensorBoard and callbacks  
- Build dense, CNN, RNN, and LSTM models  
- Convert raw text into numerical representations  
- Apply embeddings, attention, and transformers  
- Prevent overfitting and save production-ready models

---

## Concept Quick Reference

| Concept | Definition | Key Use Case |
| :---- | :---- | :---- |
| TensorBoard | TensorBoard is a TensorFlow tool used to visualize metrics, model graphs, etc. | Visualize loss and accuracy metrics for a model for evaluation. |
| Autoencoder | An autoencoder is a neural network that compresses data into more compact representations, then returns it to the data’s original state. | Denoising, dimensionality reduction, anomaly detection, etc. |
| Reconstruction Loss | Compressing data into fewer dimensions causes data loss that is evident after reconstruction. | Quantify the difference between the reconstructed output and the original input. |
| Latent Space | The compact representation of encoded data. This layer is known as the “Bottleneck”. | Captures essential features, removes noise and redundant data. |
| Backpropagation | Backpropagation is a technique used to calculate gradients throughout the layers in a Neural Network. It is performed starting at the output layer, utilizing the chain rule to move backwards throughout the network. | Used to update weights at each layer of a Neural Network utilizing the gradient. This is how the Neural Network “learns”. |
| Gradient Descent | An algorithm used to iterably update a model’s weights by moving in the path opposite of the gradient. This is the action taken at each step of backpropagation, updating the weights. | Minimize loss in your model’s prediction.  |
| Learning Rate | The Learning Rate controls the magnitude of the step taken in Gradient Descent. | Control how fast or slow your model learns. Too high or low could cause your model to struggle with converging. |
| Batch Normalization | Standardizes the inputs of a layer within a mini-batch. Mean=0 Variance=1. | Solves the internal covariate shift problem, where the distributions of each layer's inputs are changed due to weight updates. |
| Tokenization | The process of breaking text down into smaller pieces to be fed into a Neural Network. | First step in converting text to numbers (for NLP), as Neural Networks can only understand numbers.  |
| BPE (Byte-Pair Encoding) | Sub-word tokenization method. Begins with character level tokenization, then creates sub-word tokenizations utilizing the most frequent combination of characters. | Efficient tokenization, handling OOV words very efficiently. |
| One-Hot Encoding | A vectorization method, the vector is the length of the vocabulary, with a single 1 in the position of which word it is. Highly sparse and words do not have semantic similarity. | Still used in multi class classification models. |
| Word Embedding | Dense, continuous values, representing words in a high dimensional space. | Most NLP tasks |
| Word2Vec (Skip-gram/CBOW) | Word2Vec produces word embedding with 2 main architectures. Skip-Gram takes a single word and predicts its context, CBOW takes the context and predicts a single word. | Producing word embeddings |
| RNN | Recurrent Neural Network processes sequential data by utilizing a hidden state that passes through each time step and is updated with a recurrent connection. | Process sequential data, keeping a context over short sequences. |
| LSTM | Long Short Term Memory maintains a long term memory (Cell State) and short term memory (Hidden State) to process sequential data. The use of the cell state allows the model to maintain a larger context window, solving the problem of vanishing/exploding gradients. It consists of a Forget Gate, Input Gate, and Output gate. | Process sequential data, solve many issues of RNNs such as vanishing or exploding gradients. |
| GRU | Gated Recurrent Unit. In between an RNN and LSTM, having 2 gates, update and reset gate.  | A stepping stone from RNN to LSTM |
| Vanishing Gradient | Gradients disappear over longer sequences causing learning to slow or completely stop. | Ways to prevent: Consider using other activation functions such as ReLU, upgrade from RNN to LSTM. |
| Sequence Masking | Inform the model which part of the input sequence to ignore.  | If sequences are padded, the model should not learn on the pad tokens, sequence masking tells the model to ignore them. |
| Early Stopping | A form of regularization that halts the training process as soon as the model's performance on a validation set begins to decline, even if the training loss is still decreasing. | Training computationally expensive deep learning models where the optimal number of epochs is unknown. To combat overfitting. |
| Dropout | Randomly deactivate (set to 0\) a percentage of neurons during training | Reducing overfitting in large, dense layers |
| L1 Regularization | Adds a penalty proportional to the absolute value of the weights | Used for its sparsity property to act as a feature selection tool, simplifying the model |
| L2 Regularization | Adds a penalty proportional to the square of the weights  | General weight decay to keep weights small |
| Data Augmentation | Increasing data diversity by applying transformations (rotations, synonyms, etc) | Training models with limited datasets |
| Attention Mechanism | Allows the model to focus on specific parts of the input sequence | Neural Machine Translation (NMT) |
| Self-Attention | Relates different positions of a single sequence to compute a representation | Understanding context within a sentence |
| Query (Q) | Represents the current token looking for information | Matching against “Keys” to find relevance |
| Key (K) | Represents the “index” or “label” of all tokens in the sequence | Determining how much “Value” to give a token |
| Value (V) | Represents the actual information/content held by a token | Creating the weighted sum for the final output |
| Multi-Head Attention | Running multiple self-attention mechanisms in parallel | Capturing different types of relationships (grammar vs meaning etc.) |
| Transformer | An architecture relying entirely on attention without recurrence | State-of-the-art NLP (GPT, BERT) |
| Positional Encoding | Vector added to embeddings to provide info about word order | Replacing RNNs which naturally process order |
| Layer Normalization | Normalizes the activations across the features for each sample | Stabilizing training in deep Transformer blocks |
| Residual Connection | Adds the input of a block to its output (x \+ f(x)) | Preventing vanishing gradients in very deep nets |
| Feed-Forward Network (FFN) | Two linear layers with a non-linear activation (ReLU) in between | Processing the information gathered by attention |

---

## Pros & Cons

### Optimizer Comparison

| Optimizer | Pros | Cons | Best For |
| :---- | :---- | :---- | :---- |
| SGD | Low memory; theoretical convergence. | Slow; gets stuck in local minima/saddle points | Simple models; fine-tuning |
| Adam | Fast convergence; adaptive learning rates | Computationally heavy; can generalize poorly | Large datasets; Transformer models |
| RMSprop | Handles non-stationary objectives well | Requires tuning of the decay rate | RNNs and sequence modeling |

### Text Encoding Methods

| Method | Pros | Cons | Best For |
| :---- | :---- | :---- | :---- |
| One-Hot Encoding | Simple; no assumptions about relationships | High dimensionality; “sparsity”; no semantics | Small vocabularies; categorical data |
| Word Embeddings | Captures semantic similarity; dense vectors | Requires large data to train from scratch | Domain-specific text tasks |
| Pretrained Embeddings | Uses “transfer learning”; works with small data | Large memory footprint; fixed vocabulary | General NLP |

### Sequential Models

| Architecture | Pros | Cons | Best For |
| :---- | :---- | :---- | :---- |
| Simple RNN | Computationally inexpensive; fast | Vanishing gradient problem; short memory | Very short sequences |
| LSTM | Solves vanishing gradient; long-term memory | Complex (4 gates); slower to train | Long-form text; complex dependencies |
| GRU | Faster than LSTM; fewer parameters | Slightly less expressive than LSTM | Real-time apps; medium-length sequences |

### Regularization Techniques

| Technique | Pros | Cons | Best For |
| :---- | :---- | :---- | :---- |
| Dropout | Prevents co-adaptation; very effective | Increases training time; harder to tune | Deep Neural Networks (Dense layers) |
| L1 Regularization | Produces sparse models (feature selection) | Can be unstable; not differentiable at zero | Models where feature selection is needed |
| L2 Regularization | Prevents extreme weights; stable | Does not result in true sparsity | General weight decay; avoiding overfitting |
| Early Stopping | Zero computational cost; very intuitive | Might stop before reaching global optimum | Most DL models to save time/compute |

### Attention vs RNN

| Aspect | RNN-based | Attention-based | Best For |
| :---- | :---- | :---- | :---- |
| Long-range dependencies | Poor (gradient decay) | Excellent (direct connections) | Long documents |
| Training speed | Slow (sequential processing) | Fast (parallel processing) | Large-scale pretraining |
| Interpretability | Low (hidden states are opaque) | High (attention maps/weights) | Explaining model decisions |
| Memory efficiency | Efficient (O(n)) | Expensive O(n^2) | Long sequence trade offs |

### Transformer Architectures

| Architecture | Pros | Cons | Best For |
| :---- | :---- | :---- | :---- |
| Encoder-only (BERT) | Bi-directional context; great understanding | Not designed for text generation | NLU: Classification, NER, QA |
| Decoder-only (GPT) | Excellent at creative text generation | Uni-directional (cannot see the future) | NLG: Chatbots, Storytelling |
| Encoder-Decoder (T5)  | Flexible; handles input/output shifts well | High parameter count; slower inference | Translation, Summarization |

---

## When to Use What

### Choosing a Text Encoding

| If you have... | And you need... | Then use... | Because... |
| :---- | :---- | :---- | :---- |
| Small vocabulary | Simplicity | **One-Hot Encoding** | Simplest way to convert words to numerical vectors, manageable vector size due to small vocabulary |
| Large vocabulary | Semantic meaning | **Word Embeddings** | Much smaller embedding size, semantic relationships between words |
| Limited training data | Transfer learning | **Pre-trained embeddings** | With limited data, learning good embeddings from scratch is impossible. Pre-trained embeddings provide semantic knowledge from massive corpora, enabling transfer learning. |

### Choosing a Sequential Model

| If your sequences are... | And you need... | Then use... | Because... |
| :---- | :---- | :---- | :---- |
| Short (\<10 tokens) | Simple architecture | **Simple RNN** | The vanishing gradient isn't a major issue yet, and it’s the fastest to compute. |
| Long (50+ tokens) | Long-term memory | **LSTM** | Its Gating Mechanism (input, forget, output) effectively protects the gradient over long distances. |
| Variable length | Efficient training | **GRU** | It offers similar performance to LSTM but with fewer parameters and a simpler structure (no separate cell state). |

### Handling Overfitting

| If you observe... | Then try... | Because... |
| :---- | :---- | :---- |
| Train acc high, val acc low | **Dropout or More data (if available)** | Randomly deactivating neurons reduces co-adaptation between neurons, and more data improves generalization |
| Large weight values | **L1 and L2 Regularization** | L2 penalizes large weights, and L1 forces irrelevant weights to 0 |
| Validation loss increasing | **Early stopping** | Prevents training from going on too long, preventing a continued increase in loss |

### Choosing Attention vs Transformer

| If you need... | And your constraint is... | Then use... | Because... |
| :---- | :---- | :---- | :---- |
| Sequence-to-sequence with context | Moderate compute | Attention (with RNN/LSTM) | Attention enhances context modeling while keeping computational cost lower than full Transformers |
| Best performance on NLP | Sufficient GPU memory | Transformer | Transformers use self-attention and parallel preprocessing to achieve state-of-the-art NLP performance |
| Very long sequences (\>512) | Memory limited | RNN / LSTM | Transforms scale quadratically with sequence length, while RNNs scale linearly and use less memory |
| Real-time inference | Latency critical | GRU or small RNN | RNN-based models have lower inference latency and smaller memory footprints than Transforms |

---

## Essential Commands

### TensorBoard (Monday)

\# Set up TensorBoard callback

tensorboard\_callback \= tf.keras.callbacks.**TensorBoard**(**log\_dir**\='./logs')

\# Launch TensorBoard (in terminal)

\# tensorboard \--logdir=**./logs**

### Autoencoder Architecture (Monday)

\# Encoder

encoder \= keras.Sequential(\[

    layers.Dense(**128**, activation='**relu**', input\_shape=(**784**,)),

    layers.Dense(**32**, activation='**relu**'),  \# Latent space

\])

\# Decoder

decoder \= keras.Sequential(\[

    layers.Dense(**128**, activation='**relu**'),

    layers.Dense(**784**, activation='**sigmoid**'),  \# Reconstruct original

\])

### Backpropagation (Tuesday)

\# Manual gradient computation with GradientTape

with tf.GradientTape() as tape:

    predictions \= model(x\_train)

    loss \= loss\_fn(y\_train, predictions)

gradients \= tape.**gradients**(loss, model.**trainable\_variables**)

optimizer.apply\_gradients(zip(gradients, model.**trainable\_variables**))

### Batch Normalization (Tuesday)

model \= keras.Sequential(\[

    layers.Dense(64, activation='**relu**'),

    layers.**BatchNormalization()**,  \# Add batch normalization

    layers.Dense(32, activation='**relu**'),

\])

### Tokenization (Wednesday)

\# Word-level tokenization

tokenizer \= tokenizer()

tokenizer.fit\_on\_texts(texts)

sequences \= tokenizer.**texts\_to\_sequences**(texts)

\# Access vocabulary

vocab\_size \= len(tokenizer.**word\_index**)

word\_index \= tokenizer.**word\_index**

### BPE/Subword Tokenization (Wednesday)

\# Byte-Pair Encoding for handling OOV words

\# Breaks words into subword units

\# Example: "unhappiness" → \["un", "happiness"\]

\# Common libraries: SentencePiece, Hugging Face tokenizers

### Word2Vec Models (Wednesday)

\# Skip-gram: Predicts **context** words from **centerword**

\# CBOW: Predicts the **center** word from **context** words

\# Word arithmetic

\# king \- man \+ woman ≈ **queen**

\# paris \- france \+ italy ≈ **rome**

### Word Embeddings (Wednesday)

\# Keras Embedding layer

model.add(layers.Embedding(

    input\_dim=**vocab\_size**,      \# Vocabulary size

    output\_dim=**128**,     \# Embedding dimension

    input\_length=**max\_len**    \# Sequence length

))

### RNN/LSTM (Thursday)

\# Simple RNN

model.add(layers.SimpleRNN(units=**64**, activation='**tanh**'))

\# LSTM (preferred for long sequences)

model.add(layers.LSTM(units=**64**, return\_sequences=**True**))

### Sequence Padding (Thursday)

from tensorflow.keras.preprocessing.sequence import \_\_\_\_\_

padded \= **pad\_sequences**(sequences, maxlen=**max\_len**, padding='**post**', truncating='**post**')

### Sequence Masking (Thursday)

\# Masking tells model to ignore padded values

model.add(layers.Embedding(

    input\_dim=vocab\_size,

    output\_dim=embedding\_dim,

    mask\_zero=**True**  \# Treat 0 as padding

))

\# Or use explicit Masking layer

model.add(layers.Masking(mask\_value=**0**))

\#\#\# Saving & Loading Models (Friday)

\`\`\`python

\# Save entire model (architecture \+ weights)

model.save('model.h5')  \# **HDF5** format

model.save('model\_dir')  \# **SavedModel** format (recommended)

\# Load model

loaded\_model \= tf.keras.models.**load\_model**(‘**model\_dir** )

\# Save/load only weights

model.save\_weights('weights.h5')

model.load\_weights('weights.h5')

### Model Checkpoints (Friday)

checkpoint \= tf.keras.callbacks.ModelCheckpoint(

    filepath='model\_{epoch:02d}\_{**val\_loss**:.2f}.h5',

    monitor='**val\_loss**',

    save\_best\_only=**True**,

    mode='**min**'  \# 'min' for loss, 'max' for accuracy

)

\#\#\# Early Stopping (Friday)

\`\`\`python

early\_stopping \= tf.keras.callbacks.EarlyStopping(

    monitor='**val\_loss**',

    patience=**5**,

    restore\_best\_weights=**True**

)

### Regularization (Friday)

\# Dropout layer

model.add(layers.Dropout(**0.3**))

\# L2 Regularization in Dense layer

model.add(layers.Dense(64, kernel\_regularizer=tf.keras.regularizers.**12**(**0.001**)))

### Attention Mechanism (Friday)

\# Scaled Dot-Product Attention formula

\# Attention(Q, K, V) \= softmax(Q · K^T / sqrt(**d\_k**)) · V

\# Self-Attention: Q, K, V all from same input

query \= layers.Dense(**embed\_dim**)(inputs)

key \= layers.Dense(**embed\_dim**)(inputs)

value \= layers.Dense(**embed\_dim**)(inputs)

\# Compute attention scores

scores \= tf.matmul(query, key, transpose\_b=**True**)

scores \= scores / tf.math.sqrt(tf.cast(d\_model, tf.float32))

attention\_weights \= tf.nn.**softmax**(scores, axis=-1)

output \= tf.matmul(attention\_weights, **value**)

### Multi-Head Attention (Friday)

\# Using Keras built-in MultiHeadAttention

attention \= layers.MultiHeadAttention(

    num\_heads=**num\_heads**,           \# Number of parallel attention heads

    key\_dim=**d\_model // num\_heads**,             \# Dimension per head (d\_model / num\_heads)

    dropout=**0.1**              \# Dropout rate for attention weights

)

\# Self-attention: query and value are the same

output \= attention(**inputs** , **inputs**)  \# (query, value)

### Transformer Encoder Block (Friday)

\# Encoder block structure:

\# Input → MultiHeadAttention → Add & Norm → FFN → Add & Norm → Output

class TransformerEncoderBlock(keras.layers.Layer):

    def \_\_init\_\_(self, d\_model, num\_heads, ff\_dim, dropout\_rate=0.1):

        super().\_\_init\_\_()

        self.attention \= layers.MultiHeadAttention(

            num\_heads=num\_heads,

            key\_dim=d\_model // num\_heads

        )

        self.ffn \= keras.Sequential(\[

            layers.Dense(**ff\_dim**, activation='**relu**'),  \# Expand (typically 4x d\_model)

            layers.Dense(**d\_model**),                       \# Project back to d\_model

        \])

        self.layernorm1 \= layers.LayerNormalization(**epsilon=1e-6**)

        self.layernorm2 \= layers.LayerNormalization(**epsilon=1e-6**)

    

    def call(self, x):

        attn\_output \= self.attention(x, x)

        x \= self.layernorm1(x \+ **attn\_output**)  \# Residual connection

        ffn\_output \= self.ffn(x)

        return self.layernorm2(x \+ **ffn\_output**)  \# Residual connection

### Positional Encoding (Friday)

\# Why needed: Transformers have no built-in order information

\# Solution: Add position encoding to embeddings

\# Learnable positional embeddings

max\_seq\_len \= 100

d\_model \= 512

\# Token embedding \+ position embedding

token\_embed \= layers.Embedding(vocab\_size, d\_model)(inputs)

positions \= tf.range(max\_seq\_len)

pos\_embed \= layers.Embedding(**seq\_length**,  **d\_model**)(positions)

final\_embed \= token\_embed \+ **pos\_embed**

---

## Common Gotchas

| Topic | Wrong | Right |
| :---- | :---- | :---- |
| Backprop | Not detaching gradients when needed | \- It computes gradients by applying the chain rule backward through the network \- It detach tensors when gradients shouldn’t flow, e.g., for frozen layers or intermediate calculations |
| Batch Norm | Placing batch norm after activation | It is place before activation to normalize the output and keep in stable range. |
| Tokenization | Not handling OOV (out-of-vocabulary) words | Sub-word tokenization, Unknown Token(UNK) and Character level tokenization is used to handle OOV. |
| Embeddings | Using randomly initialized embeddings on small data | Utilize pre-trained Embeddings (e.g., from models like Word2Vec, GloVe, or BERT) that have been pre-trained on massive amounts of text data, for small dataset |
| RNN | Using RNN for very long sequences | RNN struggles with vanishing/exploding gradient problems so it uses LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units) to solve this with gates, allowing them to learn long-term dependencies effectively. |
| LSTM | Forgetting to set return\_sequences for stacked LSTMs | Set return\_sequences=True for all LSTM layers except the last one in a stack |
| Padding | Padding with zeros without masking | Masking to ignore padded tokens is a technique used to ensure that the padding tokens do not influence the model's computations or learning |
| Overfitting | Adding more layers when already overfitting | Should not add more layers when already overfitting instead implement Dropout or use Early Stopping because more layers just increase the model's ability to memorize noise, worsening overfitting. |
| Early Stopping | Setting patience too low | Setting the patience too low in early stopping prematurely halts model training, causing it to miss the true optimal performance and potentially resulting in an underfit model so we need to start with moderate value and observe training. |
| Saving Models | Only saving weights, not architecture | Save full model (weights \+ architecture) for easy reloading |
| Attention | Not scaling dot product by √d\_k | Scale dot product by 1/√d\_k to stabilize gradients and to normalize variance. |
| Multi-Head | Using wrong key\_dim (should be d\_model/num\_heads) | Set key\_dim \= d\_model / num\_heads so attention heads partition the representation space correctly |
| Positional Encoding | Forgetting to add position info to embeddings | Add positional encodings to embeddings so Transformers know token order |
| Transformer | Not using residual connections | Use residual (skip) connections to stabilize deep Transformer training  |
| Layer Norm | Using Batch Norm instead of Layer Norm in Transformers | Use Layer Normalization, which works independently of batch size and is sequence-friendly |
| FFN | Not expanding dimension in feed-forward (should be 4x) | Expand FFN dimension to \~4x d\_model, then project back to d\_model for better representations |

---

## Key Formulas

### Backpropagation (Chain Rule) \- THE CORE FORMULA

∂L/∂w \= ∂L/∂a × ∂a/∂z × ∂z/∂w

Where:

\- L \= Loss \- **The overall error or cost (how far off the pred was from the actual)** (loss function)

\- a \= **Output of the neuron after activation function on z (a= σ(z))** (activation output)

\- z \= **Combination of inputs and weights before activation** (weighted sum before activation)

\- w \= A parameter multiplied by the input that is adjusted to reduce loss (weight being updated)

Forward:  Input → z \= Σ(w·x) \+ b → a \= **activation(z)** → Output → Loss

Backward: Loss → ∂L/∂a → ∂a/∂z → ∂z/∂w → **∂L/∂w (weight update)**

### Gradient Descent Update

w\_new \= w\_old \- α × **∂L/∂w**

Where:

\- α \= learning rate

\- ∂L/∂w \= **gradient of the loss with respect to the weight**

### Momentum Update (SGD with Momentum)

v\_t \= β × v\_{t-1} \+ (1 \- β) × **g\_t (gradient at time step t)**

w\_new \= w\_old \- α × **v\_t (velocity at time step t)**

Where:

\- β \= momentum coefficient (typically **0.9**)

\- v \= velocity (accumulated gradient)

### L2 Regularization Loss

Total Loss \= Original Loss \+ λ × Σ(w²)

Where:

\- λ **(lambda)** \= **hyperparameter that controls how much the model penalizes large weight values to prevent overfitting** (regularization strength)

\- Σ(w²) \= **Total Loss** (sum of squared weights)

### Dropout (Training vs Inference)

Training:   output \= input × mask / (1 \- p)

Inference:  output \= input × **1**

Where:

\- p \= **Probability that a neuron is set to zero (**dropout probability)

\- mask \= random binary mask (1s and 0s)

### Embedding Lookup

Vocabulary Size: V \= **User defined size of vocabulary (e.g., 10,000)**

Embedding Dim:   D \= **User defined embedding dim (e.g., 300\)**

Embedding Matrix: V × D

Lookup: word\_index → row **word\_index** of embedding matrix → vector of dim D  
