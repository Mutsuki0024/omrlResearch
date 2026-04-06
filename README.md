# OMRL Research: Offline Meta Reinforcement Learning with World Models and Decision Transformer

This project explores how an agent can **quickly adapt to new tasks using only pre-collected data**, without interacting with the environment during training.

To understand the problem intuitively:

- Imagine training a robot to reach different target positions  
- Each target corresponds to a different task  
- Normally, the robot needs to **try repeatedly (trial-and-error)** to learn each new task  

However, in many real-world scenarios:

- Interaction is **expensive** (e.g., robotics, autonomous driving)  
- Or even **impossible** (e.g., offline datasets only)  

---

## ❓ What is the challenge?

We want an agent that can:

- Learn from a **fixed dataset (offline data)**  
- Handle **multiple different tasks**  
- **Adapt quickly** when facing a new task  

This is difficult because:

- The agent **does not know which task it is solving**  
- It **cannot collect new data to explore**  
- Data distribution may differ between training and testing  

---

## 💡 What does this project do?

This project proposes a solution by combining two ideas:

### 1. Learn to infer the task (World Model)

Instead of being told the task explicitly, the agent:

- Observes a short sequence of past behavior  
- Infers a hidden variable `z` that represents the task  

---

### 2. Learn a general decision policy (Decision Transformer)

Once the task is inferred:

- The agent uses a sequence model (similar to GPT)  
- It predicts actions based on:
  - past states
  - past actions
  - rewards
  - desired future return
  - inferred task `z`

👉 Intuition:  
"Given the situation and inferred task, what should I do next?"

---

## Overall Idea

The system works as follows:

1. Observe a small amount of past experience  
2. Infer the current task  
3. Use a general policy to act under that task  

This allows the agent to:

- Reuse past knowledge  
- Avoid retraining from scratch  
- Adapt quickly to new situations  

---

## Key Goal

> Build a system that can **generalize across tasks and adapt quickly**,  
> using **only offline data**, without additional environment interaction.

---

---

## Overview

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

## Method

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

## Installation

    git clone https://github.com/Mutsuki0024/omrlResearch.git
    cd omrlResearch

    conda create -n omrl python=3.9
    conda activate omrl

    pip install -r requirements.txt

---

## Current Experiments Environment

**PointRobot-v0**

- State: 2D position  
- Goal: sampled from `[-1, 1]^2`  
- Reward:

    r = -||s - g||_2

---

## Results

- Fast adaptation to unseen tasks  
- Stable performance in the offline setting  

![meta_dt_switch_12345](https://github.com/user-attachments/assets/4877c7aa-7b54-40f0-abf9-dbee64bfec68)

Performance is influenced by:

- context length  
- dataset quality  
- return scaling  

---

## Key Insights

- Latent task inference improves generalization  
- Decision Transformer is effective for offline RL  
- Context window size is critical  
- Distribution shift remains a challenge  

---

## Future Work

- In-episode task switching  
- Uncertainty-aware task inference  
- Improved robustness to distribution shift  
- Scaling to more complex environments  
