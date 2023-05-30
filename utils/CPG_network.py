import numpy as np
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng

rng = default_rng()
rvs = stats.uniform(-1, 2).rvs
# import

def RK45(state, A, dt):
    A1 = np.matmul(A, state)
    A2 = np.matmul(A , (state + dt / 2 * A1))
    A3 = np.matmul(A , (state + dt / 2 * A2))
    A4 = np.matmul(A , (state + dt * A3))
    return state +dt / 6 * (A1 + 2 * (A2 + A3) + A4)

def rand_CPG_network(num_dofs, inter_con_density: float = 0.5):
    inter_connection = np.triu(random(num_dofs, num_dofs, density=inter_con_density,
                                      random_state=rng, data_rvs=rvs).A, 1)
    oscillator_connection = random(1, num_dofs, density=1,
                                      random_state=rng, data_rvs=rvs).A
    oscillator_connection = np.insert(oscillator_connection, range(1, num_dofs), 0)
    weight_matrix = np.diag(oscillator_connection, 1)
    weight_matrix[::2, ::2] = inter_connection
    weight_matrix -= weight_matrix.T
    return weight_matrix


class CPG_network():
    def __init__(self, weights, dt):
        self.dt = dt
        self.n_weights = len(weights)
        self.state_shape = (self.n_weights*2, int(1))
        weights_A = np.insert(weights, range(1, self.n_weights), 0)
        wx_wy = np.diag(weights_A, 1)
        wy_wx = np.diag(-weights_A, -1)
        self.A = wx_wy+wy_wx
        self.weight_map = np.nonzero(np.triu(self.A))
        self.weights = self.A[np.nonzero(np.triu(self.A))]
        self.initial_state = np.ones(self.state_shape) * np.sqrt(2) / 2
        self.y = self.initial_state

    def set_weights(self, weights):
        A_new = np.zeros_like(self.A)
        A_new[self.weight_map] = weights
        A_new -= A_new.T
        self.A = A_new
        self.weights = weights
        self.n_weights = len(weights)

    def reset_controller(self):
        self.y = self.initial_state

    def set_initial_state(self, initial_state):
        self.initial_state = np.array(initial_state).reshape(self.state_shape)

    def set_A(self, matrix: np.array):
        assert (self.A.shape == matrix.shape), f"Matrix size does not match: {self.A.shape} vs {matrix.shape}"
        self.A = matrix
        self._update_weight_map()
        self.weights = self.A[self.weight_map]

    def update_CPG(self):
        y_t = RK45(self.y[:, -1], self.A, self.dt).reshape(self.state_shape)
        self.y = np.hstack((self.y, y_t))
        return np.array(y_t[1::2])

    def _update_weight_map(self):
        self.weight_map = np.nonzero(np.triu(self.A))
