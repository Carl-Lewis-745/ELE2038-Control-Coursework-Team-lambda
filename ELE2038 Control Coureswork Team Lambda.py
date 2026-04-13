import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==============================
# SYSTEM MATRICES
# ==============================
A = np.array([
    [0,        1,        0,         0],
    [132.995, -16.079,  20.685,     0],
    [0,       -3.678,  -15158.822,  0],
    [33.333,   0,        0,       -33.333]
])

B = np.array([0, 0, 6.889, 0])
C = np.array([0, 0, 0, 1])

# ==============================
# CONTROLLER PARAMETERS
# ==============================
Kp = 1.8461e4
Ki = 5.521e3
Kd = 0.676
Kaw = 8

V0 = 4.8617e4

# ==============================
# PID CONTROLLER
# ==============================
class PID:
    def __init__(self):
        self.int_e = 0
        self.prev_e = 0
        self.prev_t = None

    def compute(self, e, t, Vmin, Vmax, V0):
        if self.prev_t is None:
            dt = 1e-6
        else:
            dt = max(t - self.prev_t, 1e-6)

        self.prev_t = t

        de = (e - self.prev_e) / dt
        self.prev_e = e

        V_unsat = Kp*e + Ki*self.int_e + Kd*de + V0
        V = np.clip(V_unsat, Vmin, Vmax)

        # Anti-windup
        self.int_e += (e + Kaw*(V - V_unsat)) * dt

        return V

# ==============================
# NONLINEAR SYSTEM (CORRECTED)
# ==============================
def nonlinear(t, x, pid, x_ref, disturbance=False, sat=False, m=0.462, b=10.4):
    pos, vel, i, xm = x

    # ---- SAFE CURRENT LIMIT ----
    i = np.clip(i, -50, 50)

    # Constants
    g = 9.81
    k = 1885
    c = 6.811e-3
    delta = 0.65
    d = 0.42
    phi = np.deg2rad(41)
    L0 = 0.125
    L1 = 0.0241
    alpha = 1.2
    R = 2200
    tau = 0.03

    # ---- ONLY singularity protection ----
    y = max(delta - pos, 0.05)

    # Control
    e = x_ref - xm

    if sat:
        V = pid.compute(e, t, 4.7e4, 5.0e4, V0)   # tight limits → visible saturation
    else:
        V = pid.compute(e, t, -1e6, 1e6, V0)

    # Disturbance
    Fd = 5 if (disturbance and t > 2) else 0

    # Dynamics
    x_dot = vel

    v_dot = (5/(7*m))*(
        m*g*np.sin(phi)
        - k*(pos - d)
        - b*vel
        + c*i**2/(y**2)
        + Fd
    )

    L = L0 + L1*np.exp(-alpha*y)
    Ldot = alpha*L1*np.exp(-alpha*y)*vel

    i_dot = (V - R*i - i*Ldot)/L
    xm_dot = (pos - xm)/tau

    return [x_dot, v_dot, i_dot, xm_dot]

# ==============================
# LINEAR CLOSED LOOP
# ==============================
def linear(t, z, pid, x_ref):
    z = np.array(z)

    y = C @ z
    e = x_ref - y

    u = pid.compute(e, t, -1e6, 1e6, 0)

    dz = A @ z + B * u
    return dz

# ==============================
# INITIAL CONDITIONS
# ==============================
x0 = [0.5, 0, 22.1, 0.5]
t = np.linspace(0, 5, 1000)

# ==============================
# FIGURE 2: STEP RESPONSE
# ==============================
pid1 = PID()
pid2 = PID()

sol_lin = solve_ivp(linear, [0,5], x0, args=(pid1,0.52), t_eval=t)
sol_non = solve_ivp(nonlinear, [0,5], x0, args=(pid2,0.52), t_eval=t)

plt.figure()
plt.plot(sol_lin.t, sol_lin.y[-1], label="Linear")
plt.plot(sol_non.t, sol_non.y[-1], label="Nonlinear")
plt.legend()
plt.title("Figure 2: Step Response")
plt.xlabel("Time (Seconds /s)")
plt.ylabel("Position / m")
plt.grid()
plt.show()

# ==============================
# FIGURE 3: DISTURBANCE (constant force)
# ==============================
pid = PID()

sol = solve_ivp(nonlinear, [0,6], x0,
                args=(pid,0.5,True,True),
                t_eval=np.linspace(0,6,1000))

plt.figure()
plt.plot(sol.t, sol.y[-1])
plt.axvline(2, linestyle='--')
plt.title("Figure 3: Disturbance (constant force)")
plt.xlabel("Time (Seconds /s)")
plt.ylabel("Position / m")
plt.grid()
plt.show()

# ==============================
# FIGURE 4: SATURATION
# ==============================
pid1 = PID()
pid2 = PID()

sol1 = solve_ivp(nonlinear, [0,5], x0,
                 args=(pid1,0.52,False,False),
                 t_eval=t)

sol2 = solve_ivp(nonlinear, [0,5], x0,
                 args=(pid2,0.52,False,True),
                 t_eval=t)

plt.figure()
plt.plot(sol1.t, sol1.y[-1], label="No Saturation")
plt.plot(sol2.t, sol2.y[-1], label="With Saturation")
plt.legend()
plt.title("Figure 4: Saturation Effect")
plt.xlabel("Time (Seconds /s)")
plt.ylabel("Position / m")
plt.grid()
plt.show()

# ==============================
# FIGURE 5: FREQUENCY DOMAIN ROBUSTNESS
# ==============================
import control as ctrl

# --- Build state-space system ---
A_ss = A
B_ss = B.reshape(-1,1)
C_ss = C.reshape(1,-1)
D_ss = np.array([[0]])

sys = ctrl.ss(A_ss, B_ss, C_ss, D_ss)
G = ctrl.ss2tf(sys)

# --- Define PID in Laplace domain ---
s = ctrl.tf("s")
C_pid = Kp + Ki/s + Kd*s

# --- Open-loop and sensitivity ---
L = C_pid * G
S = 1 / (1 + L)
T = L / (1 + L)

# --- Frequency range ---
omega = np.logspace(-2, 5, 1000)

# --- Open-loop frequency response ---
mag, phase, omega = ctrl.freqresp(L, omega)
mag = mag.flatten()
phase = phase.flatten()

plt.figure()
plt.semilogx(omega, 20*np.log10(abs(mag)))
plt.title("Figure 5: Open-Loop Magnitude |L(jω)|")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude (dB)")
plt.grid()

plt.figure()
plt.semilogx(omega, np.degrees(phase))
plt.title("Figure 5: Open-Loop Phase ∠L(jω)")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Phase (deg)")
plt.grid()

# --- Sensitivity functions ---
magS, _, _ = ctrl.freqresp(S, omega)
magT, _, _ = ctrl.freqresp(T, omega)

plt.figure()
plt.semilogx(omega, 20*np.log10(abs(magS.flatten())), label="S (Sensitivity)")
plt.semilogx(omega, 20*np.log10(abs(magT.flatten())), label="T (Complementary)")
plt.title("Figure 5: Sensitivity Functions")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid()

plt.show()

# ==============================
# FIGURE 6: MONTE CARLO
# ==============================
plt.figure()

for _ in range(20):
    pid = PID()

    # vary parameters
    m_rand = 0.462 * (1 + 0.1*np.random.randn())
    b_rand = 10.4 * (1 + 0.1*np.random.randn())

    sol = solve_ivp(
        lambda t, x: nonlinear(t, x, pid, 0.52, False, True, m_rand, b_rand),
        [0,5],
        x0,
        t_eval=t
    )

    plt.plot(sol.t, sol.y[-1], alpha=0.3)

plt.title("Figure 6: Monte Carlo")
plt.xlabel("Time (Seconds /s)")
plt.ylabel("Position / m")
plt.grid()
plt.show()