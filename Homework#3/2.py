import matplotlib.pyplot as plt
import numpy as np
P=np.linspace(0.6,1,20000)
X = []
Y = []
for u in P:

    m = 0.5
    for n in range(2000):
        m=(u*m)*(1-m)*4

    for n in range(100):
        m=(u*m)*(1-m)*4
        Y.append(m)
        X.append(u)

plt.figure(figsize=(12, 8))
plt.plot(X, Y, ',k', alpha=0.1)
plt.title("Bifurcation Diagram")
plt.xlabel("r")
plt.ylabel("x")
plt.xlim([0.6, 1.0])
plt.ylim([0, 1])
plt.tight_layout()
plt.show()