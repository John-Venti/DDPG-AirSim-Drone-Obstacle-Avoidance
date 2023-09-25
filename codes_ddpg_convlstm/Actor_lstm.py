import tensorflow as tf
import numpy as np
from keras.layers import *

class Actor:
    def __init__(self, sess, action_bound, action_dim, state_shape, lr=1e-4, tau=0.001):
        self.sess = sess
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_shape = state_shape
        self.tau = tau
        
        self.state = tf.placeholder(tf.float32, [None, 5, state_shape])
        self.img = tf.placeholder(tf.float32, [None, 5, 64, 64, 1])

        self.post_state = tf.placeholder(tf.float32, [None, 5, state_shape])
        self.post_img = tf.placeholder(tf.float32, [None, 5, 64, 64, 1])
        self.Q_gradient =  tf.placeholder(tf.float32, [None, action_dim])
        
        with tf.variable_scope("actor"):
            self.eval_net = self._build_network(self.state, self.img, "eval_net")
            # target net is used to predict action for critic
            self.target_net = self._build_network(self.post_state, self.post_img, "target_net")
        
        self.eval_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor/eval_net")
        self.target_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor/target_net")
        
        # use negative Q gradient to guide gradient ascent
        self.policy_gradient = tf.gradients(ys=self.eval_net, xs=self.eval_param, grad_ys=-self.Q_gradient)
        self.train_step = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.policy_gradient, self.eval_param))
        
        self.update_ops = self._update_target_net_op()
        
    def _build_network(self, X, image, scope):
        with tf.variable_scope(scope):
            init_w1 = tf.truncated_normal_initializer(0., 3e-4)
            init_w2 = tf.random_uniform_initializer(-0.05, 0.05)
            dropout = 0.1
            pool = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
            ### image branch
            lstm1 = ConvLSTM2D(filters=8,kernel_size=3, strides=(1,1),padding='valid',
                                activation='relu', dropout=dropout, recurrent_dropout=2 * dropout,
                                return_sequences=True, input_shape=(None,5, 64, 64,1))(image)
            pool1 = TimeDistributed(pool)(lstm1)
            time1 = TimeDistributed(Conv2D(8, 3, padding='SAME',activation='relu'))(pool1)
            lstm2 = ConvLSTM2D(filters=8, kernel_size=3, strides=(1, 1), padding='valid',
                               activation='relu', dropout=dropout, recurrent_dropout=2 * dropout,
                               return_sequences=True)(time1)
            pool2 = TimeDistributed(pool)(lstm2)
            time2 = TimeDistributed(Conv2D(8, 3, padding='SAME',activation='relu'))(pool2)
            flatten = TimeDistributed(Flatten())(time2)
            # d = TimeDistributed(Dense(1))(flatten)
            # action = Dense(units=1, activation='tanh')(flatten)

            ### state branch
            t_lstm1 = LSTM(units=16, input_shape=(None, 5, self.state_shape), return_sequences=True)(X)
            # t_pool1 = TimeDistributed(pool)(t_lstm1)
            t_time1 = TimeDistributed(Dense(16,activation='relu'))(t_lstm1)


            concat = tf.concat([flatten, t_time1], 2)
            action_normal = LSTM(units=1, activation='tanh')(concat)

            # action_normal = tf.layers.dense(inputs=concat, units=self.action_dim, activation=tf.nn.tanh, kernel_initializer=init_w2)
            action = tf.multiply(action_normal, self.action_bound)
        return action
        
    def act(self, state):
        imgs, dstates = self._seperate_image_act(state)
        imgs = np.reshape(imgs,(1, 5, 64, 64, 1))
        dstates = np.reshape(dstates, (1, 5, 1))
        action = self.sess.run(self.eval_net, feed_dict={self.state:dstates, self.img:imgs})[0]
        return action
        
    def predict_action(self, states):
        imgs, dstates = self._seperate_image(states)
        pred_actions = self.sess.run(self.eval_net, feed_dict={self.state:dstates, self.img:imgs})
        return pred_actions
        
    def target_action(self, post_states):
        imgs, dstates = self._seperate_image(post_states)
        actions = self.sess.run(self.target_net, feed_dict={self.post_state:dstates, self.post_img:imgs})
        return actions
        
    def train(self, Q_gradient, states):
        imgs, dstates = self._seperate_image(states)
        self.sess.run(self.train_step, feed_dict={self.state:dstates, self.img:imgs, self.Q_gradient:Q_gradient})
        
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

    def _seperate_image_act(self, states):
        images = np.array([state[0] for state in states])
        # images1 = np.array([state[0] for state_his in states for state in state_his])
        dstates = np.array([state[1] for state in states])
        return images, dstates