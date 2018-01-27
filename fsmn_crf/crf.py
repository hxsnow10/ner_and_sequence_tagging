import tensorflow as tf
import numpy as np
class CRF(object):
    '''
    This is the crf class and the ner.py will user it.
    '''
    def __init__(self, batch_size, num_classes, num_steps, targets, length, targets_transition):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.targets = targets
        self.length = length
        self.targets_transition = targets_transition

    def __call__(self, logits):
        tags_scores = tf.reshape(logits, [self.batch_size, self.num_steps, self.num_classes])
        transitions = tf.get_variable("transitions", [self.num_classes + 1, self.num_classes + 1])

        dummy_val = -1000
        class_pad = tf.Variable(dummy_val * np.ones((self.batch_size, self.num_steps, 1)), dtype=tf.float32)
        observations = tf.concat([tags_scores, class_pad], 2)

        begin_vec = tf.Variable(np.array([[dummy_val] * self.num_classes + [0] for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32)
        end_vec = tf.Variable(np.array([[0] + [dummy_val] * self.num_classes for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32)
        begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_classes + 1])
        end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_classes + 1])

        observations = tf.concat([begin_vec, observations, end_vec], 1)

        mask = tf.cast(tf.reshape(tf.sign(self.targets),[self.batch_size * self.num_steps]), tf.float32)

        # point score
        point_score = tf.gather(tf.reshape(tags_scores, [-1]), tf.range(0, self.batch_size * self.num_steps) * self.num_classes + tf.reshape(self.targets,[self.batch_size * self.num_steps]))
        point_score *= mask

        # transition score
        trans_score = tf.gather(tf.reshape(transitions, [-1]), self.targets_transition)

        # real score
        target_path_score = tf.reduce_sum(point_score) + tf.reduce_sum(trans_score)

        # all path score
        total_path_score, max_scores, max_scores_pre  = self.forward(observations, transitions, self.length)

        # loss
        loss = - (target_path_score - total_path_score)

        return loss, max_scores, max_scores_pre

    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat([transitions] * self.batch_size, 0), [self.batch_size, 6, 6])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, 6, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, 6, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, 6])
            alpha_t = previous + current + transitions
            if is_viterbi:
                max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                max_scores_pre.append(tf.argmax(alpha_t, dimension=1))
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, 6, 1])
            alphas.append(alpha_t)
            previous = alpha_t           
            
        alphas = tf.reshape(tf.concat(alphas, 0), [self.num_steps + 2, self.batch_size, 6, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), 6, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, 6, 1])

        max_scores = tf.reshape(tf.concat(max_scores, 0), (self.num_steps + 1, self.batch_size, 6))
        max_scores_pre = tf.reshape(tf.concat(max_scores_pre, 0), (self.num_steps + 1, self.batch_size, 6))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre        
