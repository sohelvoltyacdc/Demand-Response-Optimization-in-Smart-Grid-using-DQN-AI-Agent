# ==============================================================================
# PROJECT: Demand Response Optimization in Smart Grid using Reinforcement Learning
# INSTITUTION: Department of EEE, Uttara University, Dhaka, Bangladesh
# DEVELOPED BY: Md. Nayeem Sarker, Md. Sohel Hossain, Abu Hanif, Md. Anisur Rahman
# SUPERVISED BY: Siam Ahmmed (Lecturer, Dept. of EEE)
# ==============================================================================
# Demand Response Optimization in Smart Grid using Reinforcement Learning

This repository contains the official open-source codebase and simulation framework for the undergraduate thesis project developed at the **Department of EEE, Uttara University, Dhaka, Bangladesh**.

## 📌 Project Overview
This research addresses the critical peak-demand shortfalls and structural grid stress in Dhaka's urban power distribution network (managed by BPDB, DESCO, and DPDC). We model the commercial and residential demand response (DR) framework as a Markov Decision Process (MDP) and implement a Deep Q-Network (DQN) AI Agent to automate optimal energy management. 

### Key Contributions:
*   **Custom Environment:** Developed `DhakaDemandResponseEnv`, a customized simulation environment built on top of the OpenAI Gymnasium interface.
*   **Stochastic Modeling:** Integrated a localized Time-of-Use (ToU) tariff scheme and a Bernoulli outage model ($p_{shed} = 0.20$) to simulate actual local load-shedding events.
*   **Performance:** Achieved an **18.5% daily cost reduction** and successfully lowered the Peak-to-Average Ratio (PAR) down to **1.38** while maintaining consumer comfort constraints[cite: 1].

---

## 👥 Authors & Research Team
*   **Md. Nayeem Sarker** (Student ID: 2231171108)[cite: 1]
*   **Md. Sohel Hossain** (Student ID: 2231171104)[cite: 1]
*   **Abu Hanif** (Student ID: 2231171101)[cite: 1]
*   **Md. Anisur Rahman** (Student ID: 2231171106)[cite: 1]

**Supervised By:**
*   **Siam Ahmmed**, Lecturer, Department of EEE, School of Science and Engineering, Uttara University[cite: 1].

---

## 🛠️ Software Stack & Dependencies
The simulation framework is implemented in Python and relies on the following core libraries[cite: 1]:
*   `python >= 3.10`
*   `torch` (PyTorch Deep Learning Engine)[cite: 1]
*   `gymnasium` (OpenAI Gym Interface)[cite: 1]
*   `numpy` (Matrix and State Operations)[cite: 1]
*   `matplotlib` (Data Visualization and Graph Plotting)[cite: 1]

---

## 🚀 How to Run the Simulation

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
