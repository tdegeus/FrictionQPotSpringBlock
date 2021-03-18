import numpy as np
from FrictionQPotSpringBlock import Line1d

def uniform(shape):
    return np.ones(shape)

N = 3
system = Line1d.System(N, uniform)

system.set_dt(0.1)
system.set_eta(2.0 * np.sqrt(3.0) / 10.0)
system.set_m(1.0)
system.set_mu(1.0)
system.set_k_neighbours(1.0)
system.set_k_frame(1.0 / float(N))
system.set_x_frame(0.0)

system.advanceRightElastic(0.2)

x = (0.5 - 0.1) * np.ones([N])

assert np.allclose(x, system.x())
