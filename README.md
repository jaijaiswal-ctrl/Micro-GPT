# Micro-GPT: Character-Level Transformer 

This project is a from-scratch implementation of a Decoder-only Transformer, similar to the architecture used in the GPT series. The goal was to build a generative AI that learns the patterns, grammar, and style of a given text file.

##  Implementation Details
I built this using **PyTorch** and optimized it specifically for **Apple Silicon (M-series chips)**. Instead of using a pre-made library, I implemented the core components of the Transformer architecture.

### Key Features:
- **Multi-Head Self-Attention:** Allows the model to understand context and relationships between characters.
- **Positional Embeddings:** Helps the model track the order of characters in a sequence.
- **MPS Acceleration:** Integrated `torch.backends.mps` to leverage the GPU on the MacBook Pro for high-speed training.
- **Stable-Mode Training:** Implemented a CPU-to-GPU batch transfer pipeline to prevent memory crashes (OOM).

## 🚀 How to Run the Project

### 1. Setup
Clone this repository and install the required library:
```bash
pip install torch

### Training
Place your training text (e.g., a book or a collection of scripts) in a file named input.txt and run the trainer:
                 python3 gpt.py
This will train the model and save the "brain" as gpt_brain.pth

### Chatting
Once the brain is saved, you can chat with your AI instantly without training:
                            python3 chat.py
Results
Trained on a dataset of Shakespearean plays, the model successfully learned the dramatic structure of a script, including character names and old-English dialogue.

### Hardware Used
Device: MacBook Pro (Apple Silicon)
Acceleration: PyTorch MPS (Metal Performance Shaders)
 
---

### Step 3: Save and Upload
1. **Save the file** (`Cmd + S`).
2. Now you have your `README.md`! 
3. Go back to your GitHub browser window and **upload this file** along with the others.

### Why this is important:
When a recruiter or another developer clicks on your GitHub link, they don't want to read your code first—they want to read the `README`. 

**By including this, you are telling them:**
*   **What you built:** A Transformer.
*   **How you built it:** PyTorch + MPS.
*   **That you solved problems:** You mentioned "Stable-Mode Training" to fix memory crashes. This shows you are a problem-solver.
*   **How to use it:** You provided clear instructions.

**Now you are 100% ready to upload to GitHub! Let's get that project live!** 🚀🌟
