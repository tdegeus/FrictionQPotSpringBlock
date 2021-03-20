import h5py
import numpy as np
import QPot
import FrictionQPotSpringBlock

with h5py.File('Load_recover.h5', 'r') as data:

    N = data["/meta/N"][...]
    alpha = data["/meta/alpha"][...]
    beta = data["/meta/beta"][...]
    QPot.random.seed(data["/meta/seed"][...])
    gamma = QPot.random.GammaList(alpha, beta)

    system = FrictionQPotSpringBlock.Line1d.System(N, gamma)

    incs = data["/stored"][...]
    redraw = data["/redraw/inc"][...]
    iredraw = 0;

    for inc in incs:

        # set position used at time of redraw, if needed
        while True:
            if iredraw == redraw.size:
                break
            if redraw[iredraw] > inc:
                break

            assert system.set_x(data["/redraw/{0:d}".format(iredraw)][...])
            iredraw += 1

        # the system is now fully determined, and deterministic
        assert not system.set_x(data["/x/{0:d}".format(inc)][...])
        assert np.allclose(system.yieldLeft(), data["/yieldLeft/{0:d}".format(inc)][...])
        assert np.allclose(system.yieldRight(), data["/yieldRight/{0:d}".format(inc)][...])
