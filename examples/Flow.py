import FrictionQPotSpringBlock.Line1d as model
import numpy as np
import tqdm

try:
    import matplotlib.pyplot as plt

    plot = True
except ImportError:
    plot = False

N = 1000
xdelta = 1e-3
eta = 0.1

system = model.System_Cuspy_Laplace(
    m=1.0,
    eta=eta,
    mu=1.0,
    k_interactions=1.0,
    k_frame=1.0,
    dt=0.1,
    shape=[N],
    seed=0,
    distribution="random",
    parameters=[2.0],
    offset=-50,
)

nstep = 2000
f = np.empty([nstep], dtype=float)

pl_v = []
pl_f = []

for v_frame in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.6, 1.9, 2.2]:
    for step in tqdm.tqdm(range(nstep)):
        system.flowSteps(n=100, v_frame=v_frame)
        f[step] = np.mean(system.f_frame)

    pl_v.append(v_frame)
    pl_f.append(np.mean(f[-100:]))

if plot:
    fig, ax = plt.subplots()
    ax.plot(pl_v, pl_f, marker=".")
    ax.plot(pl_v, eta * np.array(pl_v), marker=".")
    plt.show()
