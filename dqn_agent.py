"""
dqn_agent.py — Deep Q-Network Agent
Uttara University EEE BSc Thesis
"""
import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQNAgent:
    """
    Architecture: Input(8) -> FC(64,ReLU) -> FC(128,ReLU) -> FC(64,ReLU) -> Output(7)
    Features: Experience Replay + Target Network + Huber Loss + Adam
    """
    def __init__(self, state_size=8, action_size=7):
        self.state_size  = state_size
        self.action_size = action_size
        # Replay buffer
        self.memory      = deque(maxlen=10000)
        # Hyperparameters
        self.gamma       = 0.95    # discount factor
        self.epsilon     = 1.00    # exploration start
        self.eps_min     = 0.01    # exploration minimum
        self.eps_decay   = 0.995   # decay per episode
        self.lr          = 0.001   # Adam learning rate
        self.batch_size  = 32
        # Build online + target networks
        self.model        = self._build_model()
        self.target_model = self._build_model()
        self.update_target()
        # Logging
        self.loss_history = []

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64,  activation='relu', input_shape=(self.state_size,), name='fc1'),
            tf.keras.layers.Dense(128, activation='relu', name='fc2'),
            tf.keras.layers.Dense(64,  activation='relu', name='fc3'),
            tf.keras.layers.Dense(self.action_size, activation='linear', name='output'),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=tf.keras.losses.Huber()   # robust to outlier rewards
        )
        return model

    def update_target(self):
        """Hard update: copy weights from online to target network"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)      # explore
        q = self.model.predict(state.reshape(1,-1), verbose=0)
        return int(np.argmax(q[0]))                        # exploit

    def replay(self):
        """Sample minibatch, compute TD targets, train online network"""
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states      = np.array([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch])
        rewards     = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones       = np.array([b[4] for b in batch])
        # Current Q-values
        q_current = self.model.predict(states, verbose=0)
        # Target Q-values (from frozen target network)
        q_target_next = self.target_model.predict(next_states, verbose=0)
        for i in range(self.batch_size):
            if dones[i]:
                q_current[i][actions[i]] = rewards[i]
            else:
                q_current[i][actions[i]] = rewards[i] + self.gamma * np.max(q_target_next[i])
        hist = self.model.fit(states, q_current, epochs=1, verbose=0)
        loss = hist.history['loss'][0]
        self.loss_history.append(loss)
        # Decay epsilon
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay
        return loss

    def get_q_values(self, state):
        return self.model.predict(state.reshape(1,-1), verbose=0)[0]

    def save(self, path="dqn_model.keras"):
        self.model.save(path)
        print(f"Model saved → {path}")

    def load(self, path="dqn_model.keras"):
        self.model = tf.keras.models.load_model(path)
        self.update_target()
        print(f"Model loaded ← {path}")
