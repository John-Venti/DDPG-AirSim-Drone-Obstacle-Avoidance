import tensorflow as tf
import numpy as np
from keras.layers import *

class Critic:
    def __init__(self, sess, state_shape, action_dim, minibatch_size, lr=1e-3, tau=0.001):
        self.sess = sess
        self.tau = tau
        self.minibatch_size = minibatch_size
        
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.td_target = tf.placeholder(tf.float32, [None, 1])
        
        # input for Q network
        self.state = tf.placeholder(tf.float32, [None, 5, state_shape])
        self.img = tf.placeholder(tf.float32, [None, 5, 64, 64, 1])
        self.action = tf.placeholder(tf.float32, [None, action_dim])
        
        #input for target network
        self.t_state = tf.placeholder(tf.float32, [None, 5, state_shape])
        self.t_img = tf.placeholder(tf.float32, [None, 5, 64, 64, 1])
        self.t_action = tf.placeholder(tf.float32, [None, action_dim])
        
        with tf.variable_scope("critic"):
            self.eval_net = self._build_network(self.state, self.action, self.img, "eval_net")
            self.target_net = self._build_network(self.t_state, self.t_action, self.t_img, "target_net")
        
        self.eval_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic/eval_net")
        self.target_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic/target_net")
        
        self.loss = tf.losses.mean_squared_error(self.td_target, self.eval_net)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.action_gradients = tf.gradients(self.eval_net, self.action)
        
        self.update_ops = self._update_target_net_op()
        
    def _build_network(self, X, action, image, scope):
        with tf.variable_scope(scope):
            init_w1 = tf.truncated_normal_initializer(0., 3e-4)
            init_w2 = tf.random_uniform_initializer(-0.05, 0.05)
            pool = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
            dropout = 0.1
            # pool = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
            ### image branch
            lstm1 = ConvLSTM2D(filters=8, kernel_size=3, strides=(1, 1), padding='valid',
                               activation='relu', dropout=dropout, recurrent_dropout=2 * dropout,
                               return_sequences=True, input_shape=(None, 5, 64, 64, 1))(image)
            pool1 = TimeDistributed(pool)(lstm1)
            time1 = TimeDistributed(Conv2D(8, 3, padding='SAME',activation='relu'))(pool1)
            lstm2 = ConvLSTM2D(filters=8, kernel_size=3, strides=(1, 1), padding='valid',
                               activation='relu', dropout=dropout, recurrent_dropout=2 * dropout,
                               return_sequences=True)(time1)
            pool2 = TimeDistributed(pool)(lstm2)
            time2 = TimeDistributed(Conv2D(8, 3, padding='SAME',activation='relu'))(pool2)
            flatten = TimeDistributed(Flatten())(time2)

            ### state branch
            t_lstm1 = LSTM(units=16, return_sequences=True)(X)
            # t_pool1 = TimeDistributed(pool)(t_lstm1)
            t_time1 = TimeDistributed(Dense(16,activation='relu'))(t_lstm1)

            ### action branch
            a_b = Dense(16,activation='relu')(action)

            concat = tf.concat([flatten, t_time1], 2)
            imandsta = LSTM(units=16,activation='relu')(concat)
            final = tf.concat([imandsta, a_b], 1)

            Q = Dense(1)(final)

        return Q
        
    def target_net_eval(self, states, actions):
        imgs, dstates = self._seperate_image(states)
        Q_target = self.sess.run(self.target_net, feed_dict={self.t_state:dstates, self.t_action:actions, self.t_img:imgs})
        return Q_target
        
    def action_gradient(self, states, actions):
        imgs, dstates = self._seperate_image(states)
        return self.sess.run(self.action_gradients, feed_dict={self.state:dstates, self.action:actions, self.img:imgs})[0]
        
    def train(self, states, actions, td_target):
        imgs, dstates = self._seperate_image(states)
        actions = actions.reshape([self.minibatch_size,1])
        feed_dict = {self.state:dstates, self.action:actions, self.td_target:td_target, self.img:imgs}
        self.sess.run(self.train_step, feed_dict=feed_dict)
        
    def _update_target_net_op(self):
        ops = [tf.assign(dest_var, (1-self.tau) * dest_var + self.tau * src_var)
               for dest_var, src_var in zip(self.target_param, self.eval_param)]
        return ops

    def _seperate_image(self, states):
        # images = np.array([state[0] for state in states])
        images = np.empty(shape=(len(states), 5, 64, 64, 1))
        for i in range(len(states)):
            images[i] = np.array([state_his[0] for state_his in states[i]])
        # images = np.array([state[0] for state_his in states for state in state_his])
        dstates = np.empty(shape=(len(states), 5, 1))
        for i in range(len(states)):
            dstates[i] = np.array([state_his[1] for state_his in states[i]])
        # dstates = np.array([state[1] for state_his in states for state in state_his])
        return images, dstates