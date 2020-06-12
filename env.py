import datetime
import numpy as np
from itertools import product
from gensim.models import word2vec


class MDP(object):

    def __init__(self, n_steps, demo_path, target, n_demonstrations=20000):
        starttime = datetime.datetime.now()
        self.n_steps = n_steps
        self.n_demonstrations = n_demonstrations
        self.demonstrations, self.location_set = self.load_demonstrations(demo_path)

        self.modes = ['stay', 'walk', 'vehicle', 'train']
        self.states, self.actions = self.init_state_action()
        self.n_states, self.n_actions = len(self.states), len(self.actions)

        self.idx_state = dict(zip(range(self.n_states), self.states))
        self.state_idx = dict(zip(self.states, range(self.n_states)))

        self.idx_action = dict(zip(range(self.n_actions), self.actions))
        self.action_idx = dict(zip(self.actions, range(self.n_actions)))

        self.feature_model = word2vec.Word2Vec.load(target)

        self.fm = self.feature_matrix()
        self.transition_matrix = self._transition_matrix()

        endtime = datetime.datetime.now()
        print('environment initialization time: {}'.format(endtime - starttime))

    def init_state_action(self):
        states = [State(t, loc) for t in range(12, 12+self.n_steps) for loc in self.location_set]
        actions = [Action(loc, mode) for loc in self.location_set for mode in self.modes]

        return states, actions

    def feature_vector(self, s, a):
        state, action = self.idx_state[s], self.idx_action[a]
        token = str(state) + str(action)

        if token in self.feature_model:
            return self.feature_model[token]
        else:
            return np.zeros(100)

    def feature_matrix(self, deep=True):

        if deep:
            return np.array([self.feature_vector(s, a) for s in range(self.n_states) for a in range(self.n_actions)])

    def transition_function(self, s, a, s_
                            ):
        s, a, s_ = self.idx_state[s], self.idx_action[a], self.idx_state[s_]
        t1, l1 = s.timestamp, s.location
        t2, l2 = s_.timestamp, s_.location
        prob = 0.
        if a.destination == l2 and int(t1) + 1 == int(t2):
            prob = 1.

        return prob

    def _transition_matrix(self):

        starttime = datetime.datetime.now()
        transition_matrix = np.zeros([self.n_states, self.n_actions, self.n_states])

        # for s, a, s_ in product(range(self.n_states), range(self.n_actions), range(self.n_states)):
        #    transition_matrix[s, a, s_] = self.transition_function(s, a, s_)

        for s, a in product(range(self.n_states), range(self.n_actions)):
            state_ = State(self.idx_state[s].timestamp+1, self.idx_action[a].destination)
            if state_ in self.state_idx:
                transition_matrix[s, a, self.state_idx[state_]] = 1.

        endtime = datetime.datetime.now()
        print('transition matrix generation time: {}'.format(endtime-starttime))

        return transition_matrix

    def step(self, s, a):
        state = self.idx_state[s]
        timestamp = state.timestamp
        action = self.idx_action[a]

        state_ = State(timestamp+1, action.destination)
        s_ = self.state_idx[state_]

        return s_

    def load_demonstrations(self, path):

        id_demo = {}
        locations = []
        count = 0

        with open(path) as f:
            for line in f:
                try:
                    count += 1
                    if count % 100000 == 0:
                        print("finish {} lines".format(count))

                    tokens = line.strip('\r\n').split(',')
                    pid = tokens[0]
                    timestamp = int(tokens[1])
                    if timestamp in range(0, 12):
                        continue
                    start = tokens[2]
                    end = tokens[3]
                    mode = tokens[4]

                    if start not in locations:
                        locations.append(start)
                    if end not in locations:
                        locations.append(end)

                    s = State(timestamp, start)
                    a = Action(end, mode)
                    episode = (s, a)

                    if pid not in id_demo:
                        demonstration = [episode]
                        id_demo[pid] = demonstration
                    else:
                        id_demo[pid].append(episode)

                    if count > self.n_demonstrations * 36:
                        break

                except():
                    print('Loading demonstrations meets error ')

        return id_demo, set(locations)


class Action(object):

    def __init__(self, destination, mode):
        self.destination = destination
        self.mode = mode

    def __repr__(self):
        return self.destination + self.mode

    def __hash__(self):
        return hash(self.destination + self.mode)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self.destination + self.mode) == hash(other.destination + other.mode)
        else:
            return False


class State(object):

    def __init__(self, timestamp, location):
        self.location = location
        self.timestamp = timestamp

    def __repr__(self):
        return str(self.timestamp) + self.location

    def __hash__(self):
        return hash(str(self.timestamp) + self.location)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(str(self.timestamp) + self.location) == hash(str(other.timestamp) + self.location)
        else:
            return False
