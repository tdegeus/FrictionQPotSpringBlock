import matplotlib.pyplot as plt
import numpy as np
import sys

ret = np.genfromtxt('SimpleQuastiStatic_historic.txt', delimiter=",")

if len(sys.argv) == 2:
    test = np.genfromtxt(sys.argv[1], delimiter=",")
    assert np.allclose(ret, test)
    print('Check successful')

fig, ax = plt.subplots()
ax.plot(ret[0, :], ret[1, :])
plt.show()
