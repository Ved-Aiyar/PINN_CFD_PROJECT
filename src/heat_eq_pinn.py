import deepxde as dde
import numpy as np

# 1. Define the 1D geometry (rod length 0 to 1)
geom = dde.geometry.Interval(0, 1)

# 2. Define the governing PDE: d²T/dx² = 0
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return dy_xx

# 3. Boundary conditions: T(0)=100, T(1)=0
bc_left = dde.DirichletBC(geom, lambda x: 100, lambda x, on_boundary: on_boundary and np.isclose(x[0], 0))
bc_right = dde.DirichletBC(geom, lambda x: 0, lambda x, on_boundary: on_boundary and np.isclose(x[0], 1))

# 4. Create training data (collocation points)
data = dde.data.PDE(
    geom, pde, [bc_left, bc_right],
    num_domain=20, num_boundary=2
)

# 5. Define the neural network (Fully connected, 3 hidden layers, 20 neurons each)
net = dde.maps.FNN([1] + [20]*3 + [1], "tanh", "Glorot uniform")

# 6. Combine PDE + NN into a model
model = dde.Model(data, net)

# 7. Compile and train
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=5000)

# 8. Predict T(x) on a uniform grid for visualization
x = np.linspace(0, 1, 100)[:, None]
T_pred = model.predict(x)

# 9. Plot result
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x, T_pred, label="PINN Prediction")
plt.plot(x, 100*(1-x), "--", label="Analytical Solution")
plt.xlabel("x")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.title("1D Steady Heat Conduction")
plt.savefig("results/heat_eq_pinn_result.png", dpi=150)
plt.show()