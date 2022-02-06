import tensorflow as tf
from pc_networks import PolicyGradientNetwork
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np

class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4,
                 layer1_size=256, layer2_size=256):

        self.gamma = gamma
        self.lr = alpha
        self.n_actions = n_actions
        self.state_m = []
        self.action_m = []
        self.reward_m = []
        self.policy = PolicyGradientNetwork(n_actions=n_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))

    def choose_action(self, obs):
        state = tf.convert_to_tensor([obs], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        return action.numpy()[0]

    def store(self, obs, action, reward):
        self.state_m.append(obs)
        self.action_m.append(action)
        self.reward_m.append(reward)

    def learn(self):
        actions = tf.convert_to_tensor(self.action_m, dtype=tf.float32)
        rewards = np.array(self.reward_m)

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_m)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
