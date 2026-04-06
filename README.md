# OMRL Research: Offline Meta Reinforcement Learning with World Models and Decision Transformer

This repository implements an **Offline Meta Reinforcement Learning (OMRL)** framework that combines a **context-aware world model** with a **Meta Decision Transformer (Meta-DT)**.

The objective is to enable **fast adaptation to new tasks using only offline data**, without additional environment interaction.

---

## 🚀 Overview

This project focuses on:

- Learning **latent task representations** from offline trajectories  
- Conditioning policy behavior on inferred task context  
- Achieving **fast adaptation across tasks**  
- Addressing **distribution shift in offline RL**

The framework consists of two key components:

1. **World Model (Task Inference)**
   - Encodes trajectory context into a latent variable `z`
   - Captures hidden task information

2. **Meta Decision Transformer**
   - Sequence model conditioned on state, action, reward, RTG, and `z`
   - Outputs action sequences

---

## 🧠 Method

### Problem Setting

- Multi-task reinforcement learning  
- Offline datasets collected from expert policies  
- Test-time adaptation to unseen tasks  

### World Model

- Input: context window of trajectories  
- Model: RNN-based encoder  
- Output: latent task embedding `z_t`  

This allows implicit task inference without explicit task labels.

### Meta Decision Transformer

Input sequence:

- states  
- actions  
- rewards  
- returns-to-go (RTG)  
- latent task embedding `z`  

Output:

- predicted next action  

### Training Pipeline

1. Train expert policies (e.g., SAC)  
2. Collect offline datasets  
3. Train the world model for task inference  
4. Train Meta-DT on sequential trajectory data  
5. Evaluate on unseen tasks  

---

## ⚙️ Installation

    git clone https://github.com/Mutsuki0024/omrlResearch.git
    cd omrlResearch

    conda create -n omrl python=3.9
    conda activate omrl

    pip install -r requirements.txt

---

## 📊 Current Experiments Environment

**PointRobot-v0**

- State: 2D position  
- Goal: sampled from `[-1, 1]^2`  
- Reward:

    r = -||s - g||_2

---

## 📈 Results

- Fast adaptation to unseen tasks  
- Stable performance in the offline setting  

![meta_dt_switch_12345](https://github.com/user-attachments/assets/4877c7aa-7b54-40f0-abf9-dbee64bfec68)

Performance is influenced by:

- context length  
- dataset quality  
- return scaling  

---

## 🔍 Key Insights

- Latent task inference improves generalization  
- Decision Transformer is effective for offline RL  
- Context window size is critical  
- Distribution shift remains a challenge  

---

## 📌 Future Work

- In-episode task switching  
- Uncertainty-aware task inference  
- Improved robustness to distribution shift  
- Scaling to more complex environments  


---

## 👤 Author

GitHub: https://github.com/Mutsuki0024
