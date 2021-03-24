import numpy as np
import time
import FrictionQPotSpringBlock
import QPot

def uniform(shape):
    return np.ones(shape)

N = 3
system = FrictionQPotSpringBlock.Line1d.System(N, uniform)

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



N = 3
seed = int(time.time())
random = QPot.random.RandList()

QPot.random.seed(seed)
system = FrictionQPotSpringBlock.Line1d.System(N, random)

n = 20
x = 100.0 * np.ones((N))
redraw = []

for i in range(20):
    r = system.set_x(float(i) * x)
    if r:
        redraw += [system.QPot().currentRedraw()]

QPot.random.seed(seed)
other = FrictionQPotSpringBlock.Line1d.System(N, random)

for i in redraw:
    other.QPot().redraw(i)

other.set_x(system.x());

assert np.allclose(system.yieldLeft(), other.yieldLeft())
