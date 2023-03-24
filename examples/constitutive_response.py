import FrictionQPotSpringBlock.Line1d as model
import matplotlib.pyplot as plt
import numpy as np
import prrng

N = 1
chunk = prrng.pcg32_tensor_cumsum_1_1(
    shape=[100],
    initstate=np.arange(N),
    initseq=np.zeros(N),
    distribution=prrng.distribution.random,
    parameters=[2.0],
    align=prrng.alignment(buffer=5, margin=50, min_margin=25, strict=False),
)
chunk -= 5

fig, ax = plt.subplots()

# Cusp

system = model.System_Cuspy_Laplace(
    m=1,
    eta=1,
    mu=1,
    k_interactions=1,
    k_frame=1,
    dt=1,
    chunk=chunk,
)

u = np.linspace(0, 10, 1000)
f = np.zeros(u.size)

for i in range(u.size):
    system.u = np.array([u[i]])
    f[i] = -system.f_potential[0]

ax.plot(u, f, c="k")

# Smooth

system = model.System_Smooth_Laplace(
    m=1,
    eta=1,
    mu=1,
    k_interactions=1,
    k_frame=1,
    dt=1,
    chunk=chunk,
)

u = np.linspace(0, 10, 1000)
f = np.zeros(u.size)

for i in range(u.size):
    system.u = np.array([u[i]])
    f[i] = -system.f_potential[0]

ax.plot(u, f, c="b")

# SemiSmooth

system = model.System_SemiSmooth_Laplace(
    m=1,
    eta=1,
    mu=1,
    kappa=1,
    k_interactions=1,
    k_frame=1,
    dt=1,
    chunk=chunk,
)

u = np.linspace(0, 10, 1000)
f = np.zeros(u.size)

for i in range(u.size):
    system.u = np.array([u[i]])
    f[i] = -system.f_potential[0]

ax.plot(u, f, c="r")

# annotations

u = np.linspace(0, 10, 1000)
ax.plot(u, np.zeros_like(u), c="k", ls="--")

plt.show()
