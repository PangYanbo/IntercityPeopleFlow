import numpy as np


def solve(mdp, Q, rewards=None):

    if rewards is None:
        rewards = mdp.reward

    rewards = rewards.reshape((mdp.n_states, mdp.n_actions))
    diff = float('inf')

    while diff > 1e-3:
        V = soft_max(Q)
        Qp = rewards + 0.9 * mdp.transition_matrix.dot(V)
        # print(rewards.shape, mdp.transition_matrix.dot(V).shape)
        # Qp = reward + np.tile(V, (1, Q.shape[1])).reshape((Q.shape[0], Q.shape[1]))
        diff = np.amax(abs(Q-Qp))
        Q = Qp
    # print('Q', Q)
    pi = compute_policy(Q).astype(np.float32)

    return pi, Q


def soft_max(x, t=1):
    """

    :param x:
    :param t:
    :return:
    """

    assert t >= 0
    x = np.asarray(x)
    if len(x.shape) == 1: x = x.reshape((1, -1))
    if t == 0: return np.amax(x, axis=1)
    if x.shape[1] == 1: return x

    def softmax_2_arg(x1, x2, t):
        '''
        Numerically stable computation of t*log(exp(x1/t) + exp(x2/t))

        Parameters
        ----------
        x1 : numpy array of shape (n,1)
        x2 : numpy array of shape (n,1)

        Returns
        -------
        numpy array of shape (n,1)
            Each output_i = t*log(exp(x1_i / t) + exp(x2_i / t))
        '''

        tlog = lambda x: t * np.log(x)
        expt = lambda x: np.exp(x / t)

        max_x = np.amax((x1, x2), axis=0)
        min_x = np.amin((x1, x2), axis=0)

        return max_x + tlog(1 + expt((min_x - max_x)))

    sm = softmax_2_arg(x[:, 0], x[:, 1], t)

    for (i, x_i) in enumerate(x.T):
        if i > 1: sm = softmax_2_arg(sm, x_i, t)
        # print(i, sm)

    return sm


def compute_policy(q, ent_wt=1.0):
    """

    :param q:
    :param ent_wt:
    :return:
    """
    v = soft_max(q)
    advantage = q - np.expand_dims(v, axis=1)
    policy = np.exp((1.0 / ent_wt) * advantage)

    assert np.all(np.isclose(np.sum(policy, axis=1), 1.0)), str(policy)

    return policy


def choose_action(state, policy, greedy=0.05):
    n_actions = policy.shape[1]

    if np.random.random() > greedy:

        action = np.random.choice(n_actions, 1, p=policy[int(state)])
        # print(state, policy[state], action)
    else:
        action = np.random.choice(n_actions, 1)

    return action
