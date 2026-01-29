# MiniGPT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**A minimal, educational implementation of GPT-style transformer language models from scratch with progressive training capabilities.**

# ![MiniGPT](miniGPT_ico.ico)


## 📋 Table of Contents

- [Overview](#overview)
- [Project Philosophy](#project-philosophy)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Training a Model](#training-a-model)
  - [Progressive Model Expansion](#progressive-model-expansion)
  - [Running Inference](#running-inference)
  - [Analyzing Models](#analyzing-models)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Download Pre-trained Models](#download-pre-trained-models)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Author](#author)

---

## 🌟 Overview

MiniGPT is a research project aimed at making the construction of small language models accessible and straightforward. This is primarily a personal learning project, but it's also available for anyone interested in understanding and building transformer-based language models from the ground up.

Built entirely from scratch using PyTorch, MiniGPT demonstrates that sophisticated AI models are not mysterious black boxes—they are understandable mathematical constructs that anyone can learn, build, and improve.

### What's New in Version 2.0

- **Progressive Training**: Start small and expand your model gradually
- **Layer Freezing**: Train specific layers while keeping others frozen
- **Model Expansion**: Seamlessly grow model capacity while preserving learned weights
- **Unified Runner**: Single interface for analysis, generation, and interactive chat
- **Enhanced Tokenization**: Improved BPE tokenizer with better special token handling
- **Checkpoint Management**: Smart resumption from previous training sessions

---

## 💡 Project Philosophy

### AI Belongs to Everyone

**Artificial intelligence is not property—it is knowledge, and knowledge must be free.**

This project is built on core principles:

**1. Transparency Over Obscurity**
- No individual or corporation has the right to monopolize artificial intelligence
- AI systems should be open to inspection, understanding, and critique
- Open implementations enable worldwide learning and innovation

**2. Access Over Gatekeeping**
- Everyone should have the ability to create and deploy their own AI systems
- Educational barriers should be lowered, not raised
- Economic constraints shouldn't determine who participates in the AI revolution

**3. Knowledge Over Control**
- Understanding AI is a fundamental skill in the 21st century
- Open implementations foster collaborative development
- Diverse perspectives lead to fairer, more beneficial AI systems

**This is AI for the people, by the people.**

---

## ✨ Key Features

### 🏗️ Core Architecture
- ✅ Complete transformer implementation with multi-head self-attention
- ✅ Positional encoding for sequence awareness
- ✅ Layer normalization and residual connections
- ✅ Configurable depth, width, and attention heads

### 🚀 Training Capabilities
- ✅ **Progressive Model Expansion**: Start with a small model and grow it incrementally
- ✅ **Selective Layer Training**: Freeze specific layers while training others
- ✅ **Automatic Checkpoint Management**: Resume training from any checkpoint
- ✅ **GPU Acceleration**: Full CUDA support for faster training
- ✅ **Real-time Monitoring**: Track loss and training progress

### 🎯 Tokenization
- ✅ Byte Pair Encoding (BPE) for efficient vocabulary
- ✅ Special tokens for chat formatting (`<|user|>`, `<|assistant|>`, `<|end|>`)
- ✅ Customizable vocabulary size (1000-10000 tokens)
- ✅ Persistent tokenizer saving and loading

### 💬 Inference Modes
- ✅ **Auto-generation**: Creative text production with customizable parameters
- ✅ **Interactive Chat**: Conversational AI with memory across turns
- ✅ **Temperature Control**: Adjust randomness from deterministic to creative
- ✅ **Token Limiting**: Prevent runaway generation

### 🔍 Analysis Tools
- ✅ Comprehensive model architecture inspection
- ✅ Parameter counting and memory estimation
- ✅ Layer-wise breakdown of model components
- ✅ Vocabulary size verification
- ✅ Training progress tracking

### 🛠️ Unified Interface
- ✅ Single `runer.py` script for all operations
- ✅ Model browsing and selection
- ✅ Automatic model folder detection
- ✅ Cross-platform compatibility (Windows, Linux, macOS)

---

## 🏛️ Architecture

MiniGPT implements a decoder-only transformer architecture similar to GPT-2:

```
Input Text → Tokenization → Token Embedding
    ↓
Positional Encoding (Learned)
    ↓
┌─────────────────────────────────┐
│  Transformer Block × N          │
│  ┌───────────────────────────┐  │
│  │ Layer Normalization       │  │
│  │         ↓                 │  │
│  │  Multi-Head Self-Attention│  │
│  │    (with Causal Masking)  │  │
│  │         ↓                 │  │
│  │  Residual Connection      │  │
│  │         ↓                 │  │
│  │ Layer Normalization       │  │
│  │         ↓                 │  │
│  │  Feed-Forward Network     │  │
│  │    (4× expansion)         │  │
│  │         ↓                 │  │
│  │  Residual Connection      │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓
Final Layer Normalization
    ↓
Output Projection → Logits → Next Token Prediction
```

### Key Components

**1. Multi-Head Self-Attention**
- Scaled dot-product attention with query, key, and value projections
- Causal masking prevents the model from seeing future tokens
- Multiple attention heads capture diverse linguistic patterns

**2. Feed-Forward Networks**
- Two-layer MLP with ReLU activation
- 4× expansion in the hidden layer for increased capacity

**3. Positional Encoding**
- Learned positional embeddings (not sinusoidal)
- Enables the model to understand token positions in sequences

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended for training)
- 2GB+ RAM for small models, 8GB+ for larger models

### Step-by-Step Setup

**1. Clone the repository:**
```bash
git clone https://github.com/AbdelkaderHazerchi/minigpt.git
cd minigpt
```

**2. Create a virtual environment (recommended):**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

**3. Install dependencies:**
```bash
# Install PyTorch (with CUDA 11.8 support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install tokenizers library
pip install tokenizers
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True  # (or False if no GPU)
```

---

## 🚀 Quick Start

### Training Your First Model

**1. Prepare your data:**
```bash
mkdir data
# Add your .txt files to the data/ folder
```

**2. Start training:**
```bash
python train_2.py
```

**3. Use your model:**
```bash
python runer.py
```

That's it! The script handles tokenizer training, model initialization, and checkpoint management automatically.

---

## 📖 Usage Guide

### Training a Model

The training script (`train_2.py`) provides advanced features including progressive model expansion and selective layer training.

#### Basic Training Configuration

Edit `train_2.py` to configure your model:

```python
# Model Architecture (TARGET configuration)
TARGET_EMBED = 128      # Embedding dimension
TARGET_HEADS = 8        # Number of attention heads
TARGET_LAYERS = 2       # Number of transformer layers
TARGET_VOCAB = 4000     # Vocabulary size

# Training Settings
batch_size = 2          # Batch size (reduce if OOM)
epochs = 3500           # Total training epochs
lr = 1e-4              # Learning rate
save_every = 200        # Save checkpoint every N epochs
max_seq_len = 128       # Maximum sequence length
```

#### Progressive Training Strategy

MiniGPT supports **selective layer training**:

```python
# Train all layers
NUM_LAYERS_TO_TRAIN = None

# Train only the last 2 layers (freeze earlier layers)
NUM_LAYERS_TO_TRAIN = 2
```

This is useful for:
- Fine-tuning pre-trained models
- Gradual capacity expansion
- Efficient training with limited compute

#### Training Process

```bash
python train_2.py
```

The script will:
1. **Load or create tokenizer** from `data/*.txt` files
2. **Check for existing checkpoint** in `models/MiniGPT_LC/checkpoint.pt`
3. **Expand model if necessary** (if target config differs from checkpoint)
4. **Train the model** with real-time loss monitoring
5. **Save checkpoints** every 200 epochs
6. **Export final model** as `MiniGPT_LC.pt` when complete

#### Training Output Example

```
🔧 Using device: cuda
📋 tokenizer loaded
🆕 New model
🔓 Training all layers

📊 Statistics:
- Total parameters: 1,234,567
- Trainable parameters: 1,234,567
- Ratio: 100.00%

Epoch 0 | Loss 4.5823
Epoch 50 | Loss 2.3145
Epoch 100 | Loss 1.8234
...
💾 checkpoint saved
...
✅ Training complete — Model saved
```

---

### Progressive Model Expansion

One of MiniGPT's unique features is the ability to **expand models while preserving learned weights**.

#### How It Works

When you load a checkpoint with different dimensions than your target configuration, MiniGPT automatically:

1. **Creates a new model** with target dimensions
2. **Copies matching weights** from the old model
3. **Initializes new parameters** randomly
4. **Preserves learned knowledge** while adding capacity

#### Example: Growing a Model

**Initial training:**
```python
TARGET_EMBED = 64
TARGET_HEADS = 4
TARGET_LAYERS = 2
```

Train for 1000 epochs, then **expand the model**:

```python
TARGET_EMBED = 128      # Doubled
TARGET_HEADS = 8        # Doubled
TARGET_LAYERS = 4       # Doubled
```

Run `train_2.py` again. The script will:
- Load the smaller checkpoint
- Expand dimensions
- Continue training with the expanded model

This allows you to:
- Start training quickly with small models
- Gradually increase capacity as needed
- Avoid discarding previous learning

---

### Running Inference

The `runer.py` script provides a unified interface for all inference operations.

#### Launch the Runner

```bash
python runer.py
```

You'll see the main menu:

```
======================================================================
                        LLM MODEL MANAGER
======================================================================

Select mode:
  1. Analyze Model
  2. Generate Text
  3. Interactive Chat
  4. Exit

Enter choice (1-4):
```

#### Mode 1: Analyze Model

Provides comprehensive information about your model:

```
[MODEL INFORMATION & ANALYSIS]
  Path: models/MiniGPT_LC/MiniGPT_LC.pt
  File Size: 12.34 MB
  
[CHECKPOINT INFORMATION]
  Checkpoint Keys: model_state_dict, config, tokenizer_path
  Trained Epochs: 3500
  
[MODEL CONFIGURATION]
  Embedding Dimension: 128
  Number of Heads: 8
  Number of Layers: 2
  Max Sequence Length: 128
  Vocabulary Size: 4000
  
[PARAMETER STATISTICS]
  Total Parameters: 1,234,567
  Trainable Parameters: 1,234,567
  
[LAYER BREAKDOWN]
  head: 512,000 parameters (41.45%)
  blocks: 485,376 parameters (39.30%)
  token_emb: 512,000 parameters (41.45%)
  ...
```

#### Mode 2: Generate Text (Auto-generation)

Creative text generation with customizable parameters:

```
Available folders in 'models':
  1. MiniGPT_LC (3 models)
  2. MiniGPT_Experimental (5 models)

Select folder (or 'back'): 1

Models in 'MiniGPT_LC':
  1. MiniGPT_LC.pt (12.34 MB)
  2. checkpoint.pt (12.34 MB)

Select model (or 'back'): 1

[SUCCESS] Model loaded successfully!

==================== Auto-Generation Mode ====================

Enter your prompt: Once upon a time in a distant land

Generating...

Once upon a time in a distant land, there lived a young wizard named 
Marcus who possessed the rare ability to communicate with ancient spirits. 
His village had long forgotten the old ways, but Marcus believed that 
understanding the past was key to protecting their future...

[Generated 150 tokens]
Continue? (yes/no):
```

**Generation Parameters:**

You can adjust these in the code:
```python
# In auto_generate() function
max_len = 500        # Maximum tokens to generate
temperature = 0.8    # Sampling temperature (0.1-2.0)
```

#### Mode 3: Interactive Chat

Conversational AI with multi-turn memory:

```
==================== Interactive Chat Mode ====================

Chat with the model! Type '/quit' to exit, '/clear' to reset conversation.

You: Hello! Can you explain how transformers work?
Assistant: Transformers are significantly better than RNNs for several reasons:
1. They can process sequences in parallel, rather than sequentially
2. They don't suffer from vanishing gradient problems
3. They can capture very long-range dependencies effectively

You: /clear
[Conversation cleared]

You: Tell me a story
Assistant: In a world where code came to life, there was a small program...
```

**Special Commands:**
- `/quit` - Exit chat mode
- `/clear` - Reset conversation history

---

### Analyzing Models

Use Mode 1 to get detailed information about any model:

- File size and location
- Architecture configuration (dimensions, layers, heads)
- Parameter counts (total and trainable)
- Layer-wise parameter breakdown
- Memory estimates

This is useful for:
- Comparing different model checkpoints
- Debugging training issues
- Understanding model capacity
- Planning model expansions

---

## 📂 Project Structure

```
minigpt/
├── model.py              # Transformer architecture implementation
├── tokenizer.py          # BPE tokenizer with special tokens
├── train_2.py            # Advanced training script with model expansion
├── runer.py              # Unified runner for inference and analysis
├── data/                 # Training data directory (.txt files)
├── models/               # Saved models and checkpoints
│   └── MiniGPT_LC/       # Example model folder
│       ├── MiniGPT_LC.pt # Final trained model
│       ├── checkpoint.pt # Training checkpoint
│       └── tokenizer.json # Tokenizer vocabulary
└── README.md             # This file
```

### File Descriptions

**model.py**
- `SelfAttention`: Multi-head attention mechanism with causal masking
- `FeedForward`: Position-wise feed-forward network
- `TransformerBlock`: Single transformer layer with residual connections
- `MiniGPT`: Complete GPT model

**tokenizer.py**
- `FastBPETokenizer`: BPE tokenizer using Hugging Face tokenizers library
- Special token handling for chat formatting
- Encoding/decoding with proper byte-level processing

**train_2.py**
- Progressive training capabilities
- Automatic checkpoint resumption
- Model expansion with weight preservation
- Configurable layer freezing

**runer.py**
- Unified interface for all model operations
- Model browser and selector
- Batch processing capabilities
- Cross-platform compatibility

---

## ⚙️ Configuration

### Model Architecture Parameters

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|--------|
| `vocab_size` | Size of token vocabulary | 1000-10000 | Memory, coverage |
| `embed_dim` | Token embedding dimension | 64-512 | Model capacity |
| `num_heads` | Attention heads | 2-16 | Attention diversity |
| `num_layers` | Transformer blocks | 2-12 | Model depth |
| `max_seq_len` | Maximum sequence length | 128-2048 | Context window |

### Training Hyperparameters

| Parameter | Description | Recommended | Notes |
|-----------|-------------|-------------|-------|
| `batch_size` | Training batch size | 2-16 | Adjust based on GPU memory |
| `epochs` | Training iterations | 1000-5000 | More for complex tasks |
| `lr` | Learning rate | 1e-4 to 5e-4 | Use AdamW optimizer |
| `save_every` | Checkpoint interval | 200-500 | Balance storage vs. safety |

### Generation Parameters

| Parameter | Description | Range | Effect |
|-----------|-------------|-------|--------|
| `temperature` | Sampling randomness | 0.1-2.0 | Higher = more creative |
| `max_len` | Maximum generation length | 100-5000 | Prevent infinite loops |
| `top_k` | Top-k sampling (if implemented) | 5-50 | Limits vocabulary choices |

---

## 🔬 Technical Details

### Attention Mechanism

The self-attention computes attention scores using scaled dot-product:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

With causal masking for autoregressive generation:

```python
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float('-inf'))
```

### Training Objective

Cross-entropy loss for next-token prediction:

```python
loss = CrossEntropyLoss(logits.view(-1, vocab_size), targets.view(-1))
```

### BPE Tokenization

The tokenizer uses Byte-Pair Encoding with special tokens:

- `<pad>`: Padding token (unused in current implementation)
- `<unk>`: Unknown token for out-of-vocabulary words
- `<|user|>`: Marks the start of user messages
- `<|assistant|>`: Marks the start of assistant responses
- `<|end|>`: Signals the end of generation

### Checkpoint Format

Checkpoints are saved as PyTorch dictionaries:

```python
{
    "model_state_dict": model.state_dict(),  # Model weights
    "epoch": current_epoch,                   # Training progress
    "config": {
        "vocab_size": 4000,
        "embed_dim": 128,
        "num_heads": 8,
        "num_layers": 2,
        "max_seq_len": 128
    },
    "tokenizer_path": "tokenizer.json"       # Relative path to tokenizer
}
```

---

## 📥 Download Pre-trained Models

Ready-to-use models are available for download:

### Option 1: With Experimental Models (720MB)

Includes multiple experimental models for comparison:

**[Download MiniGPT with Experimental Models [720MB]](https://www.dropbox.com/scl/fi/ohp32tdei4k025hs0ncrj/miniGPT.rar?rlkey=usoda580lb5qcb3ypyhqd3eff&st=7mrmp2fk&dl=1)**

Contains:
- Production-ready models
- Multiple checkpoint variants
- Experimental architectures
- Complete tokenizers

### Option 2: Without Experimental Models (330.5MB)

Streamlined package with essential models only:

**[Download MiniGPT Essentials [330.5MB]](https://www.dropbox.com/scl/fi/yy5cmdj69hfoo7r2bymq5/miniGPT-nm.rar?rlkey=wxdt0gkgj7gbnw2iz43bw18ar&st=f2hyh7s7&dl=1)**

Contains:
- Core trained models
- Tokenizers
- Ready-to-use checkpoints

### After Downloading

1. Extract the archive
2. Place the `models/` folder in the project root
3. Run `python runer.py` to use the models immediately

---

## 💡 Best Practices

### For Training

**1. Data Quality**
- Use clean, well-formatted text data
- Remove excessive whitespace and special characters
- Ensure consistent encoding (UTF-8)

**2. Vocabulary Size**
- Small datasets (< 1MB): 2000-4000 tokens
- Medium datasets (1-10MB): 4000-6000 tokens
- Large datasets (> 10MB): 6000-10000 tokens

**3. Model Scaling**
- Start small: `embed_dim=64, num_layers=2`
- Gradually increase capacity as needed
- Use progressive expansion to preserve learning

**4. Training Duration**
- Monitor loss curves to detect convergence
- Save checkpoints frequently (every 200 epochs)
- Don't over-train (watch for overfitting)

**5. Learning Rate**
- Start with 1e-4 for stable training
- Increase to 5e-4 for faster convergence (if stable)
- Decrease to 1e-5 if training becomes unstable

### For Inference

**1. Temperature Control**
- `0.1-0.5`: Very focused, deterministic output
- `0.6-0.9`: Balanced creativity and coherence
- `1.0-1.5`: High creativity, more randomness
- `> 1.5`: Experimental, often incoherent

**2. Prompt Engineering**
- Be clear and specific in prompts
- Use conversational markers (`User:`, `Assistant:`) for chat
- Provide context when necessary

**3. Generation Limits**
- Set `max_len` to reasonable values (100-500 tokens)
- Monitor for repetition or degradation
- Use early stopping if quality decreases

**4. Model Selection**
- Use higher-epoch checkpoints for better quality
- Test multiple checkpoints to find the best
- Consider model size vs. quality trade-offs

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size in train_2.py
batch_size = 1  # Or 2, minimum

# Reduce model size
TARGET_EMBED = 64    # From 128
TARGET_LAYERS = 2    # From 4

# Reduce sequence length
max_seq_len = 64     # From 128
```

#### 2. Poor Text Quality

**Symptoms:**
- Repetitive text
- Nonsensical output
- Short, incomplete responses

**Solutions:**
- Train for more epochs (increase `epochs`)
- Use more/better training data
- Increase model capacity (`embed_dim`, `num_layers`)
- Adjust temperature (try 0.8-1.0)
- Verify tokenizer is working correctly

#### 3. Slow Training Speed

**Symptoms:**
- Each epoch takes very long
- GPU utilization is low

**Solutions:**
```python
# Verify CUDA is being used
print(torch.cuda.is_available())  # Should be True

# Increase batch size (if memory allows)
batch_size = 8  # Or higher

# Reduce max_seq_len for faster processing
max_seq_len = 64  # From 128

# Consider using mixed precision training (advanced)
```

#### 4. Tokenizer Errors

**Symptoms:**
```
KeyError: token not found
IndexError: index out of range
```

**Solutions:**
- Ensure `tokenizer.json` is in the model folder
- Retrain tokenizer if vocabulary is corrupted:
  ```bash
  rm models/MiniGPT_LC/tokenizer.json
  python train_2.py  # Will create new tokenizer
  ```
- Check vocab_size matches between model and tokenizer

#### 5. Model Not Learning

**Symptoms:**
- Loss stays high or doesn't decrease
- Output is random gibberish

**Solutions:**
- Check your training data quality
- Verify data files are in `data/` folder
- Increase learning rate to 5e-4
- Train for more epochs
- Check for data encoding issues (use UTF-8)

#### 6. Generation Hangs or Loops

**Symptoms:**
- Model keeps generating the same tokens
- Generation never stops

**Solutions:**
```python
# Set strict maximum length
max_len = 500  # Hard limit

# Check for <|end|> token generation
# Ensure tokenizer decode() respects end token

# Lower temperature for more deterministic output
temperature = 0.7  # From 1.0
```

---

## 🤝 Contributing

Contributions are welcome! This project thrives on community involvement.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
   - Follow existing code style
   - Add comments for complex logic
   - Test your changes thoroughly
4. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**
   - Describe your changes clearly
   - Reference any related issues

### Areas for Contribution

- **Tokenization Methods**: WordPiece, SentencePiece implementations
- **Training Optimizations**: Gradient accumulation, mixed precision
- **Sampling Methods**: Nucleus sampling, beam search, top-k
- **Architecture Variants**: Encoder-decoder, sparse attention
- **Evaluation Tools**: Perplexity metrics, benchmark datasets
- **Documentation**: Tutorials, examples, translations
- **UI Improvements**: Better terminal interface, web UI

### Code Style

- Use clear, descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Follow PEP 8 for Python code

---

## 📜 License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2026 Abdelkader Hazerchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📚 Citation

If you use MiniGPT in your research or project, please cite:

```bibtex
@software{minigpt2026,
  author = {Hazerchi, Abdelkader},
  title = {MiniGPT: A Minimal GPT Implementation with Progressive Training},
  year = {2026},
  url = {https://github.com/AbdelkaderHazerchi/minigpt},
  version = {2.0}
}
```

---

## 👤 Author

**Abdelkader Hazerchi**

- **GitHub**: [@AbdelkaderHazerchi](https://github.com/AbdelkaderHazerchi)
- **Email**: abdelkaderhaz96@gmail.com
- **LinkedIn**: [Abdelkader Hazerchi](https://www.linkedin.com/in/abdelkader-hazerchi-430153397/)

---

## 🙏 Acknowledgments

This project stands on the shoulders of giants:

- **Attention Is All You Need** (Vaswani et al., 2017) - The original transformer paper
- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019) - GPT-2 architecture
- The **PyTorch** team for an excellent deep learning framework
- The **Hugging Face** team for the tokenizers library
- The **open-source community** for making AI accessible to everyone

---

## 💭 Closing Thoughts

MiniGPT is more than code—it's a statement about the democratization of AI knowledge. In a world where artificial intelligence increasingly shapes our lives, understanding how these systems work should not be a privilege of the few, but a right of the many.

This project proves that:
- **AI is learnable**: You don't need a PhD to understand transformers
- **AI is buildable**: You don't need billions of dollars to create language models
- **AI belongs to everyone**: Open-source implementations enable global participation

Every person who learns from this code, every student who builds upon it, every researcher who extends it—they are all part of a movement toward a future where AI serves humanity, not corporate interests.

### Download

miniGPT-With experimental models: [Download [720MB]](https://www.dropbox.com/scl/fi/ohp32tdei4k025hs0ncrj/miniGPT.rar?rlkey=usoda580lb5qcb3ypyhqd3eff&st=7mrmp2fk&dl=1) <br>
miniGPT-Without experimental models: [Download [330.5MB]](https://www.dropbox.com/scl/fi/yy5cmdj69hfoo7r2bymq5/miniGPT-nm.rar?rlkey=wxdt0gkgj7gbnw2iz43bw18ar&st=f2hyh7s7&dl=1)

### Goals of This Project

1. **Educational**: Help people understand transformer architectures from first principles
2. **Accessible**: Provide a simple, runnable implementation anyone can experiment with
3. **Extensible**: Create a foundation that others can build upon and improve
4. **Practical**: Demonstrate that small language models can be trained on consumer hardware

### The Future

The future of AI is:
- **Open**: Transparent implementations, not black boxes
- **Collaborative**: Built by communities, not corporations
- **Distributed**: Running on everyone's hardware, not centralized clouds
- **Beneficial**: Serving all of humanity, not just shareholders

**Join us in building this future.**

---

## 🌐 Learn More

Want to dive deeper? Check out these resources:

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Original paper)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)

---

## ⭐ Star This Project

If you found MiniGPT useful, please consider starring the repository on GitHub! Your support helps others discover this project and encourages continued development.

**[⭐ Star on GitHub](https://github.com/AbdelkaderHazerchi/minigpt)**

---

## 📣 Spread the Word

Help us democratize AI knowledge:

- Share this project with students and researchers
- Write blog posts or tutorials using MiniGPT
- Create educational content around the code
- Contribute improvements and extensions
- Support open-source AI development

---

*"The only way to make AI truly beneficial is to make it truly open."*

---

**Build. Learn. Share. Repeat.**

---

© 2026 Abdelkader Hazerchi | Made with ❤️ for the open-source community
