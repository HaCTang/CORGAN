import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np
# from organ.prior_classifier import predict_molecule, load_model
# from organ.mol_metrics import decode

class ClassifyRollout(object):
    """用于条件生成的rollout策略模型"""

    def __init__(self, lstm, update_rate, pad_num, ord_dict):
        """初始化参数并定义模型架构"""
        self.lstm = lstm
        self.update_rate = update_rate
        self.pad_num = pad_num
        self.ord_dict = ord_dict
        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.learning_rate = self.lstm.learning_rate

        # # 加载分类器模型
        # self.classifier = load_model()

        # 复制rollout.py中的其他初始化代码
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit()
        self.g_output_unit = self.create_output_unit()

        # 定义placeholder
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.given_num = tf.placeholder(tf.int32)
        
        # 处理输入
        with tf.device("/cpu:0"):
            inputs = tf.split(axis=1, num_or_size_splits=self.sequence_length,
                            value=tf.nn.embedding_lookup(self.g_embeddings, self.x))
            self.processed_x = tf.stack(
                [tf.squeeze(input_, [1]) for input_ in inputs])

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))

        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                           dynamic_size=False, infer_shape=True)

        # 定义循环函数
        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            x_tp1 = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i + 1, x_tp1, h_t, given_num, gen_x

        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)
            gen_x = gen_x.write(i, next_token)
            return i + 1, x_tp1, h_t, given_num, gen_x

        i, x_t, h_tm1, given_num, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                      tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                      self.h0, self.given_num, gen_x))

        _, _, _, _, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x))

        self.gen_x = self.gen_x.stack()
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])

    def get_reward(self, sess, input_x, rollout_num, dis, reward_fn = None, D_weight=1):
        """计算奖励，结合判别器和分类器的输出"""
        reward_weight = 1 - D_weight
        rewards = []

        for i in range(rollout_num):
            already = []
            for given_num in range(1, self.sequence_length):
                feed = {self.x: input_x, self.given_num: given_num}
                outputs = sess.run([self.gen_x], feed)
                generated_seqs = outputs[0]
                gind = np.array(range(len(generated_seqs)))

                # 判别器奖励
                feed = {dis.input_x: generated_seqs, 
                        dis.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(dis.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])

                # 分类器奖励
                if reward_fn:
                    
                    ypred = D_weight * ypred
                    
                    for k, r in reversed(already):
                        generated_seqs = np.delete(generated_seqs, k, 0)
                        gind = np.delete(gind, k, 0)
                        ypred[k] += reward_weight * r

                    if generated_seqs.size:
                        rew = reward_fn(generated_seqs)
                        
                    # Add the just calculated rewards
                    for k, r in zip(gind, rew):
                        ypred[k] += reward_weight * r

                    # Choose the seqs finished in the last iteration
                    for j, k in enumerate(gind):
                        if input_x[k][given_num] == self.pad_num and input_x[k][given_num-1] == self.pad_num:
                            already.append((k, rew[j]))
                    already = sorted(already, key=lambda el: el[0])

                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # 最后一个字符的奖励
            feed = {dis.input_x: input_x, dis.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(dis.ypred_for_auc, feed)
            
            if reward_fn:
                ypred = D_weight * np.array([item[1]
                                             for item in ypred_for_auc])
            else:
                ypred = np.array([item[1] for item in ypred_for_auc])

            if i == 0:
                rewards.append(ypred)
            else:
                rewards[-1] += ypred

        rewards = np.transpose(np.array(rewards)) / \
            (1.0 * rollout_num)
        return rewards

    def create_recurrent_unit(self):
        """Defines the recurrent process in the LSTM."""

        # Weights and Bias for input and hidden tensor
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def update_recurrent_unit(self):
        """Updates the weights and biases of the rollout's LSTM
        recurrent unit following the results of the training."""

        # Weights and Bias for input and hidden tensor
        self.Wi = self.update_rate * self.Wi + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wi)
        self.Ui = self.update_rate * self.Ui + \
            (1 - self.update_rate) * tf.identity(self.lstm.Ui)
        self.bi = self.update_rate * self.bi + \
            (1 - self.update_rate) * tf.identity(self.lstm.bi)

        self.Wf = self.update_rate * self.Wf + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wf)
        self.Uf = self.update_rate * self.Uf + \
            (1 - self.update_rate) * tf.identity(self.lstm.Uf)
        self.bf = self.update_rate * self.bf + \
            (1 - self.update_rate) * tf.identity(self.lstm.bf)

        self.Wog = self.update_rate * self.Wog + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wog)
        self.Uog = self.update_rate * self.Uog + \
            (1 - self.update_rate) * tf.identity(self.lstm.Uog)
        self.bog = self.update_rate * self.bog + \
            (1 - self.update_rate) * tf.identity(self.lstm.bog)

        self.Wc = self.update_rate * self.Wc + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wc)
        self.Uc = self.update_rate * self.Uc + \
            (1 - self.update_rate) * tf.identity(self.lstm.Uc)
        self.bc = self.update_rate * self.bc + \
            (1 - self.update_rate) * tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        """Defines the output process in the LSTM."""

        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_output_unit(self):
        """Updates the weights and biases of the rollout's LSTM
        output unit following the results of the training."""

        self.Wo = self.update_rate * self.Wo + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + \
            (1 - self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_params(self):
        """Updates all parameters in the rollout's LSTM."""
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.update_recurrent_unit()
        self.g_output_unit = self.update_output_unit()