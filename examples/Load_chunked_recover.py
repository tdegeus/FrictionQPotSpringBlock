import sys
import h5py
import numpy as np
import prrng
import FrictionQPotSpringBlock
import matplotlib.pyplot as plt

assert len(sys.argv) == 2

ret = np.genfromtxt('Load.txt', delimiter=",")

with h5py.File(sys.argv[1], 'r') as data:

    N = data["/meta/N"][...]
    initstate = data["/meta/initstate"][...]
    initseq = data["/meta/initseq"][...]
    generators = prrng.pcg32_array(initstate, initseq)

    nchunk = 4000

    ymin = data["/ymin/{0:d}".format(0)][...]
    istart = data["/istart/{0:d}".format(0)][...]
    state = data["/state/{0:d}".format(0)][...]

    generators.restore(state)
    y = 2.0 * generators.random([nchunk])
    y[:, 0] = ymin
    y = np.cumsum(y, 1)

    system = FrictionQPotSpringBlock.Line1d.System(N, y)

    system.set_dt(0.1)
    system.set_eta(2.0 * np.sqrt(3.0) / 10.0)
    system.set_m(1.0)
    system.set_mu(1.0)
    system.set_k_neighbours(1.0)
    system.set_k_frame(1.0 / float(N))

    test = np.empty((2, data["/stored"].size))

    x_frame = data["/x_frame"][...]

    for inc in data["/stored"][...]:

        x = data["/x/{0:d}".format(inc)][...]

        if system.any_redraw(x):

            ymin = data["/ymin/{0:d}".format(inc)][...]
            istart = data["/istart/{0:d}".format(inc)][...]
            state = data["/state/{0:d}".format(inc)][...]

            generators.restore(state)
            y = 2.0 * generators.random([nchunk])
            y[:, 0] = ymin
            y = np.cumsum(y, 1)

            system.set_y(istart, y)

        system.set_x_frame(x_frame[inc])
        system.set_x(x)

        test[0, inc] = system.x_frame()
        test[1, inc] = np.mean(system.f_frame())

assert np.allclose(ret, test)

fig, ax = plt.subplots()
ax.plot(ret[0, :], ret[1, :])
plt.show()
