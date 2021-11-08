import os

import FrictionQPotSpringBlock.Line1d as model
import matplotlib.pyplot as plt
import numpy as np
import prrng
import tqdm

N = 1000

initstate = np.arange(N)
initseq = np.zeros(N)
generators = prrng.pcg32_array(initstate, initseq)


y = 2.0 * generators.random([20000])
y = np.cumsum(y, 1)
y -= 50.0

xdelta = 1e-3

system = model.System(N, y)

system.set_dt(0.1)
system.set_eta(2.0 * np.sqrt(3.0) / 10.0)
system.set_m(1.0)
system.set_mu(1.0)
system.set_k_neighbours(1.0)
system.set_k_frame(1.0 / N)

ninc = 1000
ret = np.empty([2, ninc])

pbar = tqdm.tqdm(total=ninc)

for inc in range(ninc):

    # Apply event-driven protocol.
    if inc == 0:
        system.set_x_frame(0.0)  # initial quench
    elif inc % 2 != 0:
        system.advanceEventRightElastic(xdelta)  # elastically advance -> mechanical equilibrium
    else:
        system.advanceEventRightKick(xdelta)  # apply kick

    # Minimise energy.
    if inc % 2 == 0:
        niter = system.minimise()
        pbar.n = inc
        pbar.set_description(f"inc = {inc:4d}, niter = {niter:8d}")
        pbar.refresh()

    # Extract output data.
    ret[0, inc] = system.x_frame()
    ret[1, inc] = np.mean(system.f_frame())


if os.path.isfile("SimpleQuastiStatic_historic.txt"):
    test = np.genfromtxt("SimpleQuastiStatic_historic.txt", delimiter=",")
    assert np.allclose(ret, test)

fig, ax = plt.subplots()
ax.plot(ret[0, :], ret[1, :])
plt.show()
