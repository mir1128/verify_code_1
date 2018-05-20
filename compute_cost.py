import numpy as np
import tensorflow as tf
from create_placeholders import create_placeholders
from initialize_parameters import initialize_parameters
from forward_propagation import forward_propagation


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y);
    cost = tf.reduce_mean(entropy)

    return cost


if __name__ == '__main__':
    tf.reset_default_graph()

    with tf.Session() as sess:
        np.random.seed(1)
        X, Y = create_placeholders(64, 64, 3, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        init = tf.global_variables_initializer()
        sess.run(init)
        a = sess.run(cost, {X: np.random.randn(4, 64, 64, 3), Y: np.random.randn(4, 6)})
        print("cost = " + str(a))
