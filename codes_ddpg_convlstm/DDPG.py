import tensorflow as tf
import numpy as np
import os
from Actor_lstm import Actor
from Critic_lstm import Critic
from OUNoise import OrnsteinUhlenbeckActionNoise
from ReplayMemory import ReplayMemory


class DDPG_agent:
    def __init__(self, sess, state_shape, action_bound, action_dim,
                 memory_size=100000, minibatch_size=5, gamma=0.99, tau=0.001, train_after=10):
        #100000 128 200
        self.actor = Actor(sess, action_bound, action_dim, state_shape,lr = 0.0001, tau=tau)
        self.critic = Critic(sess, state_shape, action_dim, minibatch_size,lr = 0.001, tau=tau)
        self.state_shape = state_shape
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.memory_size = memory_size
        self.replay_memory = ReplayMemory(self.memory_size)
        self.sess = sess
        self.minibatch_size = minibatch_size
        self.action_bound = action_bound
        self.gamma = gamma
        self.train_after = max(minibatch_size, train_after)
        self.num_action_taken = 0
        self.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_dim))

    def observe(self, state, action, reward, post_state, terminal):
        self.replay_memory.append(state, action, reward, post_state, terminal)

    def act(self, state, noise=True):
        action = self.actor.act(state)
        if noise:
            noise = self.action_noise()
            action = np.clip(action + noise, -self.action_bound, self.action_bound)[0]
        else:
            action = np.clip(action, -self.action_bound, self.action_bound)[0]
        self.num_action_taken += 1
        return action

    def update_target_nets(self):
        # update target net for both actor and critic
        self.sess.run([self.actor.update_ops, self.critic.update_ops])

    def train(self,times = 1):
        if self.num_action_taken >= self.train_after:
            for i in range(times):
                #print ("training:{} / {}".format(i,times),end = '\r')

                # 1 sample random minibatch from replay memory
                states, actions, rewards, post_states, terminals, states_his, post_states_his = \
                    self.replay_memory.sample(self.minibatch_size)

                # 2 use actor's target net to select action for Si+1, denote as mu(S_i+1)
                action = self.actor.target_action(post_states_his)
                print('action', action)

                # 3 use critic's target net to evaluate Q(S_i+1, a_i+1) and calculate td target
                Q_target = self.critic.target_net_eval(post_states_his, action)
                rewards = rewards.reshape([self.minibatch_size, 1])
                terminals = terminals.reshape([self.minibatch_size, 1])
                td_target = rewards + self.gamma * Q_target * (1 - terminals)
                print('Q', Q_target)
                # 4 update critic's online network
                self.critic.train(states_his, actions, td_target)

                # 5 predict action using actors online network and calculate the sampled gradients
                pred_actions = self.actor.predict_action(states_his)
                Q_gradients = self.critic.action_gradient(states_his, pred_actions) / self.minibatch_size

                # 6 update actor's online network
                self.actor.train(Q_gradients, states_his)

                # 7 apply soft replacement for both target networks
                self.update_target_nets()

    def save(self, saver, dir):
        path = os.path.join(dir, 'model')
        saver.save(self.sess, path)

    def load(self, saver, dir):
        path = os.path.join(dir, 'checkpoint')
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        return False
