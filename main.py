import numpy as np

import matplotlib
# matplotlib.use("Qt5Agg")  # optionally force locally

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.animation import FuncAnimation
import time

EPS = 1e-12

# ---------------------- Smoothness helpers ----------------------

def _path_length(P):
    P = np.asarray(P, float)
    if len(P) < 2:
        return 0.0
    seg = np.diff(P, axis=0)
    return float(np.sum(np.linalg.norm(seg, axis=1)))

def _unwrap_angles(theta):
    return np.unwrap(theta)

def _segment_headings(P):
    P = np.asarray(P, float)
    seg = np.diff(P, axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    return _unwrap_angles(ang), np.linalg.norm(seg, axis=1)

def _path_smoothness_metrics(P):
    """
    Returns (S_kappa2_per_m, S_HTV_per_m, L) for a polyline path P (Nx2).
    """
    P = np.asarray(P, float)
    if len(P) < 3:
        L = _path_length(P)
        return 0.0, 0.0, L

    theta, ds = _segment_headings(P)              # len N-1
    dtheta = theta[1:] - theta[:-1]               # len N-2
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi

    ds_local = 0.5 * (ds[1:] + ds[:-1])          # len N-2
    ds_local = np.where(ds_local < 1e-12, 1e-12, ds_local)

    kappa = np.abs(dtheta) / ds_local
    L = float(np.sum(ds))
    L = max(L, 1e-12)

    S_k2 = float(np.sum((kappa**2) * ds_local))
    S_k2_per_m = S_k2 / L
    S_htv_per_m = float(np.sum(np.abs(dtheta))) / L

    return S_k2_per_m, S_htv_per_m, L

# ====================== VFCBF core ======================

def heat_kernel(x, tau, beta):
    r2 = float(x @ x)
    return beta * np.exp(-r2 / (4.0 * tau))

def heat_field(x, tau, beta):
    k = heat_kernel(x, tau, beta)
    return k * x

def Dh(x, tau, beta):
    k = heat_kernel(x, tau, beta)
    I = np.eye(2)
    return k * I - (k / (2.0 * tau + EPS)) * np.outer(x, x)

def a_normal(x, tau, beta):
    """
    a(x) = Dh(x) @ h(x), with h(x) = heat_field(x)
    returns (a, h)
    """
    h = heat_field(x, tau, beta)
    return Dh(x, tau, beta) @ h, h

# This is the gaussian curve where tau is the width (robot feels the object from farther away w/ larger tau) and beta
# is the peak intensity

# ====================== Circle constraints: CBF, VFCBF, BOTH ======================

def circle_constraints_select(
    p, c, R,
    # VFCBF params
    gamma=0.28, tau=0.65, beta=1.5,
    # Distance CBF safety
    alpha_cbf=2.0,
    # geometry and activation
    margin=0.08, activate_dist=2.0,
    mode="both"
):
    """
    Returns active constraints depending on mode:
      cbf:   n^T u >= -alpha_cbf * (d - R_eff)
      vfcbf: a^T u >= -gamma * ||h||
      both:  apply both
    Also returns R_eff and optional vfcbf_data.
    """
    R_eff = R + margin #effective radius
    rel = p - c
    d = np.linalg.norm(rel) + EPS
    if d - R_eff > activate_dist:
        return [], R_eff, None

    cons = []
    vfcbf_data = None

    # scalar distance CBF
    n_hat = rel / d
    h_dist = d - R_eff
    a_safety = n_hat
    b_safety = -alpha_cbf * h_dist

    # build VFCBF around signed boundary coordinate
    p_obs = c + R_eff * n_hat
    s = d - R_eff
    x = np.sign(s) * (p - p_obs)

    a_vf, h_vf = a_normal(x, tau, beta)
    b_vf = -gamma * np.linalg.norm(h_vf)

    if mode in ("cbf", "both"):
        cons.append((a_safety, b_safety))

    if mode in ("vfcbf", "both"):
        cons.append((a_vf, b_vf))
        vfcbf_data = {'x': x, 'h_vf': h_vf, 'a_vf': a_vf}

    return cons, R_eff, vfcbf_data

# ====================== Braking distance CBF ======================

def braking_cbf_constraint(p, v, c, R_eff, a_max=1.5, alpha_brake=1.0, activate_dist=2.0):
    # might need to change a_max and alpha_brake
    """
    Kinematic braking constraint: the agent must always be slow enough
    to stop before reaching R_eff.

    Safe set:  h = 2 * a_max * (d - R_eff) - ||v||^2  >= 0
    Lie deriv: dh/dt = 2*(a_max*n_hat - v)^T u
    CBF cond:  dh/dt >= -alpha_brake * h
    =>  [2*(a_max*n_hat - v)]^T u  >=  -alpha_brake * h

    Only activated when the agent is within activate_dist of R_eff.
    Returns (a_vec, b_scalar) or None if inactive.
    """
    rel = p - c
    d = np.linalg.norm(rel) + EPS
    R_gap = d - R_eff

    if R_gap > activate_dist:
        return None

    n_hat = rel / d
    speed_sq = float(v @ v)

    h = 2.0 * a_max * R_gap - speed_sq
    a_vec = 2.0 * (a_max * n_hat - v)
    b_scalar = -alpha_brake * h

    return a_vec, b_scalar

# ====================== Turning Constraint ======================

def steering_cbf_constraint(p, v, c, R_eff, a_lat_max=3.0, alpha_steer=1.2, activate_dist=2.0):
    """
    Kinematic steering constraint: the agent must always have enough
    lateral acceleration capacity to curve around R_eff.

    Safe set:  h = a_lat_max * d^2 - 2 * R_eff * ||v||^2  >= 0
    Lie deriv (lateral mapped): pushes velocity into the tangent vector
    """
    rel = p - c
    d = np.linalg.norm(rel) + EPS
    R_gap = d - R_eff

    if R_gap > activate_dist or d <= R_eff:
        return None

    n_hat = rel / d
    speed_sq = float(v @ v)

    # 1. Calculate the Safe Set (h)
    h = a_lat_max * (d**2) - 2.0 * R_eff * speed_sq

    # 2. Find the Tangent Vector (steer left or right?)
    # We pick the direction that matches the robot's current slight drift
    t_hat = np.array([-n_hat[1], n_hat[0]])
    if v @ t_hat < 0:
        t_hat = -t_hat # Flip to match existing momentum

    # 3. Construct the CBF gradient (a_vec)
    # We map the d^2 distance gradient to the tangent vector to force steering,
    # while the velocity gradient acts as a speed limiter.
    a_vec = (2.0 * a_lat_max * d * t_hat) - (4.0 * R_eff * v)
    
    # 4. The relaxation scalar
    b_scalar = -alpha_steer * h

    return a_vec, b_scalar

#================ braking vs. steering decision ================
def agile_cbf_constraint(p, v, c, R_eff, a_max=2.5, alpha_brake=1.0, a_lat_max=3.0, alpha_steer=1.2, activate_dist=2.0):
    """
    Boolean Composition CBF: h_combined = max(h_brake, h_steer).
    Allows the robot to seamlessly switch between braking and steering
    depending on which maneuver provides a higher safety margin.
    """
    rel = p - c
    d = np.linalg.norm(rel) + EPS
    R_gap = d - R_eff

    # 1. Activation Check
    if R_gap > activate_dist or d <= R_eff:
        return None

    n_hat = rel / d
    speed_sq = float(v @ v)

    # 2. Calculate BOTH Safe Sets
    h_brake = 2.0 * a_max * R_gap - speed_sq
    h_steer = a_lat_max * (d**2) - 2.0 * R_eff * speed_sq

    # 3. Apply the "OR" logic (Choose the maximum safe set)
    if h_steer > h_brake:
        # === STEERING IS SAFER ===
        # Calculate tangent vector matching current drift
        t_hat = np.array([-n_hat[1], n_hat[0]])
        if v @ t_hat < 0:
            t_hat = -t_hat

        # Steering Gradient
        a_vec = (2.0 * a_lat_max * d * t_hat) - (4.0 * R_eff * v)
        b_scalar = -alpha_steer * h_steer
    else:
        # === BRAKING IS SAFER ===
        # Braking Gradient
        a_vec = 2.0 * (a_max * n_hat - v)
        b_scalar = -alpha_brake * h_brake

    return a_vec, b_scalar

# ====================== Projection ======================

def project_seq(u0, A, b, passes=10):
    u = u0.copy()
    if len(b) == 0:
        return u
    m = len(b)
    for _ in range(passes):
        for k in range(m):
            ak = A[k]
            bk = b[k]
            viol = ak @ u - bk
            if viol < 0.0:
                step = (bk - ak @ u) / (ak @ ak + EPS)
                u = u + step * ak
    return u

def project_qp_like(u_nom, A, b, passes=12):
    if not A:
        return u_nom.copy()

    A_mat = np.vstack(A)
    b_vec = np.asarray(b)

    if np.all(A_mat @ u_nom >= b_vec - 1e-12):
        return u_nom.copy()

    u = project_seq(u_nom, A_mat, b_vec, passes=passes)
    cand = [u]
    m = len(b_vec)
    for i in range(m):
        for j in range(i + 1, m):
            Ai, Aj = A_mat[i], A_mat[j]
            M = np.vstack([Ai, Aj])
            rhs = np.array([b_vec[i], b_vec[j]])
            try:
                lam = np.linalg.solve(M @ M.T, rhs - M @ u_nom)
                u_ij = u_nom + M.T @ lam
                if np.all(A_mat @ u_ij >= b_vec - 1e-9):
                    cand.append(u_ij)
            except np.linalg.LinAlgError:
                pass

    feas = [c for c in cand if np.all(A_mat @ c >= b_vec - 1e-9)]
    if feas:
        d = [np.linalg.norm(c - u_nom) for c in feas]
        return feas[int(np.argmin(d))]

    viol = A_mat @ u_nom - b_vec
    k = int(np.argmin(viol))
    ak = A_mat[k]
    bk = b_vec[k]
    step = (bk - ak @ u_nom) / (ak @ ak + EPS)
    return u_nom + max(0.0, step) * ak

def project_with_speed_and_repair(u_nom, A, b, vmax, strict_normals=None, passes=12, reits=4):
    u = project_qp_like(u_nom, A, b, passes=passes)
    for _ in range(reits):
        n = np.linalg.norm(u)
        if n > vmax:
            u = u * (vmax / (n + EPS))
        u = project_qp_like(u, A, b, passes=passes)
        if strict_normals:
            for (ai, bi) in strict_normals:
                viol = ai @ u - bi
                if viol < 0.0:
                    step = (bi - ai @ u) / (ai @ ai + EPS)
                    u = u + step * ai
    return u

# ====================== Nominal & smoothing ======================

def u_nominal(p, v, g, k_att=1.45, k_damp=0.42, vmax=1.05):
    u = k_att * (g - p) - k_damp * v
    n = np.linalg.norm(u)
    if n > vmax:
        u = u * (vmax / (n + EPS))
    return u

def smooth_control(u_new, u_prev, alpha=0.72):
    return (1.0 - alpha) * u_prev + alpha * u_new

# ====================== Interactive Sim ======================

class InteractiveSimDrag:
    def __init__(self):
        # Arena limits
        self.xmin, self.xmax = -2.5, 2.7
        self.ymin, self.ymax = -2.5, 2.5

        # Obstacles
        self.circ_c = np.array([0.2, 0.0])
        self.circ_R = 0.85

        # VFCBF params
        self.tau_c, self.beta_c, self.gamma_c = 0.65, 2.5, 0.4
        #Increase the repulsion (tau_c value) to mimic an autonomous vehicle reacting further away from an obstacle (originally 0.65)
        #Increasing beta_c will increase repulsive field around object and make agent react earlier (originally 1.5)
        #gamma_c is the coefficient on the barrier condition, effectively the stiffness

        # tau_c = 0.65, beta_c = 3 works but the VFCBF swings really far out from the field

        # Sim params
        self.dt = 0.028
        self.steps = 1800
        self.vmax = 2.5 # to model an autonomous vehicle, increase vmax to a higher velocity (originally 1)
        # This should affect the decleration the CBF so the VFCBF thrives
        self.alpha_u = 0.1 #originally 0.7, decrease so that the inertia is lower (underactuation)

        # Braking CBF params
        # a_max: max deceleration (m/s^2) — higher = tighter braking envelope
        # alpha_brake: CBF decay rate — higher = more aggressive correction when h < 0
        self.a_max = 2.0
        self.alpha_brake = 1.5
        self.a_lat_max = 1.3
        self.alpha_steer = 1.5

        # Vector field grid
        self.grid_Nx = 30
        self.grid_Ny = 26

        # State (single-agent fields used for "one-run" mode)
        self.goal = np.array([+2.0, +1.2], dtype=float)
        self.start = np.array([-2.0, -1.1], dtype=float)
        self.p = self.start.copy()
        self.v = np.zeros(2)
        self.u_prev = np.zeros(2)
        self.path = []

        # Dual-agent state (for "run both")
        self.p_cbf = self.start.copy()
        self.v_cbf = np.zeros(2)
        self.u_prev_cbf = np.zeros(2)
        self.path_cbf = []

        self.p_vf = self.start.copy()
        self.v_vf = np.zeros(2)
        self.u_prev_vf = np.zeros(2)
        self.path_vf = []

        self.anim = None
        self.running = False
        self.dragging_artist = None

        # Mode: cbf, vfcbf, both (vector field visualization + radio)
        self.mode = "both"

        # For single-agent animation, optionally override which constraint set is used
        self.sim_mode_override = None  # "cbf" or "vfcbf" or "both" or None

        # Are we running both agents simultaneously?
        self.running_both = False

        # Figure + axes
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)

        # Buttons (added "Run Both")
        ax_run_vf   = self.fig.add_axes([0.82, 0.02, 0.16, 0.06])
        ax_run_cbf  = self.fig.add_axes([0.64, 0.02, 0.16, 0.06])
        ax_run_both = self.fig.add_axes([0.46, 0.02, 0.16, 0.06])
        ax_reset    = self.fig.add_axes([0.28, 0.02, 0.16, 0.06])
        ax_compare  = self.fig.add_axes([0.10, 0.02, 0.16, 0.06])

        ax_mode     = self.fig.add_axes([0.08, 0.085, 0.18, 0.07])

        self.btn_run_vf   = Button(ax_run_vf,  "Run VFCBF Agent")
        self.btn_run_cbf  = Button(ax_run_cbf, "Run CBF Agent")
        self.btn_run_both = Button(ax_run_both, "Run Both (CBF & VFCBF)")
        self.btn_reset    = Button(ax_reset,   "Reset")
        self.btn_compare  = Button(ax_compare, "Compare (Open-loop)")

        # Radio buttons for mode select (vector field visualization)
        self.rad_mode = RadioButtons(ax_mode, ("cbf", "vfcbf", "both"), active=2)
        self.rad_mode.on_clicked(self.on_mode_change)

        self.btn_run_vf.on_clicked(self.on_run_vf)
        self.btn_run_cbf.on_clicked(self.on_run_cbf)
        self.btn_run_both.on_clicked(self.on_run_both)
        self.btn_reset.on_clicked(self.on_reset)
        self.btn_compare.on_clicked(self.on_compare)

        self._draw_static_scene()

        # Single-agent artists (blue)
        (self.goal_artist,)  = self.ax.plot([], [], 'rx', ms=12, mew=2.2, label="Goal", picker=8)
        (self.start_artist,) = self.ax.plot([], [], 'g^', ms=10, label="Start", picker=8)
        (self.agent_artist,) = self.ax.plot([], [], 'bs', ms=10, label="Agent (single)")
        (self.trail_artist,) = self.ax.plot([], [], 'b-', lw=1.8, alpha=0.85, label="Trajectory (single)")

        # Dual-agent artists
        (self.agent_cbf_artist,) = self.ax.plot([], [], 'ms', ms=10, label="CBF agent")
        (self.trail_cbf_artist,) = self.ax.plot([], [], 'm-', lw=2.0, alpha=0.85, label="CBF traj")

        (self.agent_vf_artist,) = self.ax.plot([], [], 'cs', ms=10, label="VFCBF agent")
        (self.trail_vf_artist,) = self.ax.plot([], [], 'c-', lw=2.0, alpha=0.85, label="VFCBF traj")

        # Vector field quiver
        self.quiv = None

        # Control arrows
        self.ctrl_quiv = None          # single
        self.ctrl_quiv_cbf = None      # dual cbf
        self.ctrl_quiv_vf = None       # dual vfcbf

        # Text artists
        self.start_coord_text = self.ax.text(0, 0, '', fontsize=9, ha='center')
        self.goal_coord_text = self.ax.text(0, 0, '', fontsize=9, ha='center')
        self.vfcbf_text = self.ax.text(
            0, 0, '', fontsize=8, color='darkred',
            bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3')
        )

        self.on_reset(None)

        # Events
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.ax.legend(loc="upper left", fontsize=8)
        plt.show()

    def _draw_static_scene(self):
        self.ax.clear()
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title("Drag Goal/Start. Radio controls vector field. Buttons run CBF/VFCBF/Both.")
        circ = plt.Circle(tuple(self.circ_c), self.circ_R, color='k', fill=False, lw=2)
        self.ax.add_patch(circ)
        self._draw_mode_banner()

    def _draw_mode_banner(self):
        txt = f"Field Mode: {self.mode.upper()}"
        self.ax.text(
            0.02, 0.97, txt, transform=self.ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2')
        )

    # --------- Vector Field computation and drawing ----------

    def _compute_vector_field(self):
        xs = np.linspace(self.xmin, self.xmax, self.grid_Nx)
        ys = np.linspace(self.ymin, self.ymax, self.grid_Ny)
        XX, YY = np.meshgrid(xs, ys)

        Ux = np.zeros_like(XX)
        Uy = np.zeros_like(YY)

        _, R_eff_at_center, _ = circle_constraints_select(
            self.circ_c + np.array([self.circ_R, 0.0]),
            self.circ_c, self.circ_R,
            mode=self.mode
        )

        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                q = np.array([XX[i, j], YY[i, j]])

                if np.linalg.norm(q - self.circ_c) < R_eff_at_center - 1e-6:
                    Ux[i, j] = 0.0
                    Uy[i, j] = 0.0
                    continue

                u_nom = u_nominal(q, np.zeros(2), self.goal, vmax=self.vmax)

                A, B = [], []
                circle_cons, _, _ = circle_constraints_select(
                    q, self.circ_c, self.circ_R,
                    gamma=self.gamma_c, tau=self.tau_c, beta=self.beta_c,
                    mode=self.mode
                )
                for a_c, b_c in circle_cons:
                    A.append(a_c)
                    B.append(b_c)

                strict = []
                if self.mode in ("cbf", "both") and circle_cons:
                    strict = [circle_cons[0]]

                u_proj = project_with_speed_and_repair(
                    u_nom, A, B, vmax=self.vmax, strict_normals=strict, passes=8, reits=2
                )

                Ux[i, j] = u_proj[0]
                Uy[i, j] = u_proj[1]

        M = np.hypot(Ux, Uy)
        M_safe = M + 1e-9
        Ux_norm = Ux / M_safe
        Uy_norm = Uy / M_safe
        rel = np.clip(M / (self.vmax + 1e-9), 0.2, 1.0)
        Ux_vis = Ux_norm * rel
        Uy_vis = Uy_norm * rel

        mask_inside = (np.hypot(XX - self.circ_c[0], YY - self.circ_c[1]) < R_eff_at_center - 1e-6)
        Ux_vis[mask_inside] = 0.0
        Uy_vis[mask_inside] = 0.0

        return XX, YY, Ux_vis, Uy_vis

    def _draw_or_update_quiver(self):
        XX, YY, Ux, Uy = self._compute_vector_field()
        if self.quiv is None:
            self.quiv = self.ax.quiver(
                XX, YY, Ux, Uy,
                angles='xy', scale_units='xy', scale=18,
                width=0.006, alpha=0.85, pivot='mid', minlength=0
            )
            self.ax.quiverkey(
                self.quiv, X=0.86, Y=1.03, U=1,
                label='Vector field dir and relative mag', labelpos='E'
            )
        else:
            self.quiv.set_UVC(Ux, Uy)

    # -------------------------------------------------------

    def on_mode_change(self, label):
        self.mode = label
        self._redraw_scene_and_readd_artists()
        self.fig.canvas.draw_idle()

    def _redraw_scene_and_readd_artists(self):
        self._draw_static_scene()

        # Re-add artists after clearing axis
        (self.goal_artist,)  = self.ax.plot([self.goal[0]], [self.goal[1]], 'rx', ms=12, mew=2.2, label="Goal", picker=8)
        (self.start_artist,) = self.ax.plot([self.start[0]], [self.start[1]], 'g^', ms=10, label="Start", picker=8)

        (self.agent_artist,) = self.ax.plot([], [], 'bo', ms=8, label="Agent (single)")
        (self.trail_artist,) = self.ax.plot([], [], 'b-', lw=1.8, alpha=0.85, label="Trajectory (single)")

        (self.agent_cbf_artist,) = self.ax.plot([], [], 'mo', ms=7, label="CBF agent")
        (self.trail_cbf_artist,) = self.ax.plot([], [], 'm-', lw=2.0, alpha=0.85, label="CBF traj")

        (self.agent_vf_artist,) = self.ax.plot([], [], 'co', ms=7, label="VFCBF agent")
        (self.trail_vf_artist,) = self.ax.plot([], [], 'c-', lw=2.0, alpha=0.85, label="VFCBF traj")

        self.start_coord_text = self.ax.text(
            self.start[0], self.start[1] + 0.15,
            f"({self.start[0]:.2f}, {self.start[1]:.2f})",
            fontsize=9, ha='center'
        )
        self.goal_coord_text = self.ax.text(
            self.goal[0], self.goal[1] - 0.2,
            f"({self.goal[0]:.2f}, {self.goal[1]:.2f})",
            fontsize=9, ha='center'
        )
        self.vfcbf_text = self.ax.text(
            0, 0, '', fontsize=8, color='darkred',
            bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3')
        )

        self.quiv = None
        self._draw_or_update_quiver()

        self.ctrl_quiv = None
        self.ctrl_quiv_cbf = None
        self.ctrl_quiv_vf = None

        self.ax.legend(loc="upper left", fontsize=8)

    def on_pick(self, event):
        if self.running:
            return
        if event.artist in (self.goal_artist, self.start_artist):
            self.dragging_artist = event.artist

    def on_motion(self, event):
        if self.running or self.dragging_artist is None or event.inaxes != self.ax:
            return
        x = float(np.clip(event.xdata, self.xmin, self.xmax))
        y = float(np.clip(event.ydata, self.ymin, self.ymax))
        self.dragging_artist.set_data([x], [y])

        coord_str = f"({x:.2f}, {y:.2f})"
        if self.dragging_artist is self.goal_artist:
            self.goal[:] = [x, y]
            self.goal_coord_text.set_position((x, y - 0.2))
            self.goal_coord_text.set_text(coord_str)
        else:
            self.start[:] = [x, y]
            self.start_coord_text.set_position((x, y + 0.15))
            self.start_coord_text.set_text(coord_str)

        self._draw_or_update_quiver()
        self.fig.canvas.draw_idle()

    def on_release(self, _event):
        self.dragging_artist = None

    def on_reset(self, _event=None):
        if self.running and self.anim:
            self.anim.event_source.stop()

        self.running = False
        self.anim = None
        self.running_both = False
        self.sim_mode_override = None

        self.btn_run_vf.label.set_text("Run VFCBF Agent")
        self.btn_run_cbf.label.set_text("Run CBF Agent")
        self.btn_run_both.label.set_text("Run Both (CBF & VFCBF)")

        self._redraw_scene_and_readd_artists()

        # Clear paths and single state
        self.path = []
        self.p = self.start.copy()
        self.v = np.zeros(2)
        self.u_prev[:] = 0.0

        self.agent_artist.set_data([], [])
        self.trail_artist.set_data([], [])

        # Clear dual state
        self.p_cbf = self.start.copy()
        self.v_cbf = np.zeros(2)
        self.u_prev_cbf = np.zeros(2)
        self.path_cbf = [self.p_cbf.copy()]

        self.p_vf = self.start.copy()
        self.v_vf = np.zeros(2)
        self.u_prev_vf = np.zeros(2)
        self.path_vf = [self.p_vf.copy()]

        self.agent_cbf_artist.set_data([], [])
        self.trail_cbf_artist.set_data([], [])
        self.agent_vf_artist.set_data([], [])
        self.trail_vf_artist.set_data([], [])

        self.fig.canvas.draw_idle()

    # ------------------- Dynamics step (shared) -------------------

    def _compute_control_step(self, p, v, u_prev, mode):
        """
        One control step for a given agent mode: "cbf" or "vfcbf" or "both".
        Returns: u (smoothed), u_proj (pre-smooth), vfcbf_data, compute_ms
        """
        t0 = time.perf_counter()

        u_nom = u_nominal(p, v, self.goal, vmax=self.vmax)
        A, B = [], []

        circle_cons, R_eff, vfcbf_data = circle_constraints_select(
            p, self.circ_c, self.circ_R,
            gamma=self.gamma_c, tau=self.tau_c, beta=self.beta_c,
            mode=mode
        )
        for a_c, b_c in circle_cons:
            A.append(a_c)
            B.append(b_c)

        # Only apply braking mode to CBF because VFCBF has build in deaccelearation
        if mode in ("cbf", "both"):
            agile_cons = agile_cbf_constraint(
                p, v, self.circ_c, R_eff,
                a_max=self.a_max, alpha_brake=self.alpha_brake,
                a_lat_max = self.a_lat_max, alpha_steer = self.alpha_steer
            )
            if agile_cons is not None:
                A.append(agile_cons[0])
                B.append(agile_cons[1])

        '''if mode in ("cbf", "both"):
            steer_cons = steering_cbf_constraint(
                p, v, self.circ_c, R_eff,
                a_lat_max = self.a_lat_max, alpha_steer = self.alpha_steer
            )
            if steer_cons is not None:
                A.append(steer_cons[0])
                B.append(steer_cons[1])'''

        strict = []
        if mode in ("cbf", "both") and circle_cons:
            strict = [circle_cons[0]]

        u_proj = project_with_speed_and_repair(
            u_nom, A, B, vmax=self.vmax, strict_normals=strict, passes=14, reits=5
        )
        u = smooth_control(u_proj, u_prev, alpha=self.alpha_u)

        t1 = time.perf_counter()
        return u, u_proj, vfcbf_data, (t1 - t0) * 1000.0

    def _repair_circle_penetration(self, p_next):
        # use "both" for repair so nobody ends inside
        _, R_eff, _ = circle_constraints_select(
            self.circ_c + np.array([self.circ_R, 0.0]),
            self.circ_c, self.circ_R,
            activate_dist=1e9, mode="both"
        )
        rel = p_next - self.circ_c
        d = np.linalg.norm(rel) + EPS
        if d < R_eff:
            return self.circ_c + (rel / d) * R_eff
        return p_next

    # ------------------- Single-run buttons -------------------

    def _start_run_single(self, sim_mode):
        if self.running:
            return
        self.running = True
        self.running_both = False
        self.sim_mode_override = sim_mode

        if sim_mode == "cbf":
            self.btn_run_cbf.label.set_text("Running...")
            self.btn_run_vf.label.set_text("Run VFCBF Agent")
        else:
            self.btn_run_vf.label.set_text("Running...")
            self.btn_run_cbf.label.set_text("Run CBF Agent")
        self.btn_run_both.label.set_text("Run Both (CBF & VFCBF)")

        self.p = self.start.copy()
        self.v = np.zeros(2)
        self.u_prev = np.zeros(2)
        self.path = [self.p.copy()]

        # hide dual artists while single runs (optional)
        self.agent_cbf_artist.set_data([], [])
        self.trail_cbf_artist.set_data([], [])
        self.agent_vf_artist.set_data([], [])
        self.trail_vf_artist.set_data([], [])

        self.ctrl_quiv_cbf = None
        self.ctrl_quiv_vf = None

        self.anim = FuncAnimation(
            self.fig, self._update_frame, frames=self.steps,
            interval=self.dt * 1000, blit=False, repeat=False
        )
        self.fig.canvas.draw_idle()

    def on_run_vf(self, _event=None):
        self._start_run_single("vfcbf")

    def on_run_cbf(self, _event=None):
        self._start_run_single("cbf")

    # ------------------- NEW: Run both -------------------

    def on_run_both(self, _event=None):
        if self.running:
            return
        self.running = True
        self.running_both = True
        self.sim_mode_override = None

        self.btn_run_both.label.set_text("Running...")
        self.btn_run_vf.label.set_text("Run VFCBF Agent")
        self.btn_run_cbf.label.set_text("Run CBF Agent")

        # init both agents
        self.p_cbf = self.start.copy()
        self.v_cbf = np.zeros(2)
        self.u_prev_cbf = np.zeros(2)
        self.path_cbf = [self.p_cbf.copy()]

        self.p_vf = self.start.copy()
        self.v_vf = np.zeros(2)
        self.u_prev_vf = np.zeros(2)
        self.path_vf = [self.p_vf.copy()]

        # hide single agent artists (optional)
        self.agent_artist.set_data([], [])
        self.trail_artist.set_data([], [])
        self.ctrl_quiv = None

        self.ctrl_quiv_cbf = None
        self.ctrl_quiv_vf = None

        self.anim = FuncAnimation(
            self.fig, self._update_frame_both, frames=self.steps,
            interval=self.dt * 1000, blit=False, repeat=False
        )
        self.fig.canvas.draw_idle()

        self.cbf_crashed = False
        self.vf_crashed = False

    # ------------------- Stop -------------------

    def _stop_simulation(self):
        if self.anim is not None:
            self.anim.event_source.stop()
        self.running = False
        self.anim = None

        self.btn_run_vf.label.set_text("Run VFCBF Agent")
        self.btn_run_cbf.label.set_text("Run CBF Agent")
        self.btn_run_both.label.set_text("Run Both (CBF & VFCBF)")

        self.running_both = False
        self.sim_mode_override = None

    def check_collision(p, circ_c, circ_R):
        """Returns True if the agent's position p is inside the obstacle."""
        distance = np.linalg.norm(p - circ_c)
        # We use circ_R (the physical radius) to trigger the 'stop'
        return distance <= circ_R

    # ------------------- Text formatting -------------------

    def _format_text_single(self, run_mode, u_proj, u, comp_time_ms, vfcbf_data):
        time_text = f"$\\mathbf{{Compute:}}$ {comp_time_ms:.2f} ms"
        if vfcbf_data:
            x_val = vfcbf_data['x']
            h_val = vfcbf_data['h_vf']
            a_val = vfcbf_data['a_vf']
            Dh_val = Dh(x_val, self.tau_c, self.beta_c)

            dh_str_row1 = f"{Dh_val[0,0]:.2f}, {Dh_val[0,1]:.2f}"
            dh_str_row2 = f"{Dh_val[1,0]:.2f}, {Dh_val[1,1]:.2f}"

            vfcbf_text_str = (
                f"$h(x) \\approx [{h_val[0]:.2f}, {h_val[1]:.2f}]$\n"
                f"$Dh(x) \\approx [[{dh_str_row1}], [{dh_str_row2}]]$\n"
                f"$a(x) \\approx [{a_val[0]:.2f}, {a_val[1]:.2f}]$\n"
                f"$x \\approx [{x_val[0]:.2f}, {x_val[1]:.2f}]$"
            )
            return vfcbf_text_str + "\n---\n" + f"{run_mode.upper()}\n" + time_text
        else:
            return f"{run_mode.upper()} active, VFCBF inactive here\n---\n" + time_text

    def _format_text_both(self, ms_cbf, ms_vf):
        return (
            "RUN BOTH\n"
            f"CBF compute:   {ms_cbf:.2f} ms\n"
            f"VFCBF compute: {ms_vf:.2f} ms"
        )

    # ------------------- Frame update: single -------------------

    def _update_frame(self, frame_num):
        run_mode = self.sim_mode_override or self.mode

        R_stop = self.circ_R + 0.08  # R_eff (same margin used in constraints)
        dist_to_obs = np.linalg.norm(self.p - self.circ_c)
        speed = np.linalg.norm(self.v)
        stalled = (dist_to_obs <= R_stop * 1.08) and (speed < 0.05)
        if dist_to_obs <= R_stop or stalled:
            reason = "COLLISION" if dist_to_obs <= R_stop else "STALLED at boundary"
            print(f"{reason}: {run_mode.upper()} agent stopped. dist={dist_to_obs:.3f}, speed={speed:.3f}")
            self._stop_simulation()
            return (self.agent_artist, self.trail_artist, self.vfcbf_text)

        if np.linalg.norm(self.p - self.goal) < 0.05:
            self._stop_simulation()
            return

        u, u_proj, vfcbf_data, ms = self._compute_control_step(self.p, self.v, self.u_prev, run_mode)
        self.u_prev = u.copy()

        p_next = self.p + self.dt * u
        p_next = self._repair_circle_penetration(p_next)

        self.v = (p_next - self.p) / (self.dt + EPS)
        self.p = p_next
        self.path.append(self.p.copy())

        # Draw single
        self.agent_artist.set_data([self.p[0]], [self.p[1]])
        path_np = np.array(self.path)
        self.trail_artist.set_data(path_np[:, 0], path_np[:, 1])

        self.vfcbf_text.set_position((self.p[0] + 0.1, self.p[1] + 0.1))
        self.vfcbf_text.set_text(self._format_text_single(run_mode, u_proj, u, ms, vfcbf_data))

        # Control arrow (single)
        if self.ctrl_quiv is None:
            self.ctrl_quiv = self.ax.quiver(
                [self.p[0]], [self.p[1]], [u[0]], [u[1]],
                angles='xy', scale_units='xy', scale=1.0,
                width=0.012, alpha=0.95, pivot='tail', color='r'
            )
            self.ax.quiverkey(self.ctrl_quiv, X=0.13, Y=1.03, U=1,
                              label='control u (single)', labelpos='E')
        else:
            self.ctrl_quiv.set_offsets(np.array([[self.p[0], self.p[1]]]))
            self.ctrl_quiv.set_UVC([u[0]], [u[1]])

        if frame_num >= self.steps - 1:
            self._stop_simulation()

        return (self.agent_artist, self.trail_artist, self.vfcbf_text)

    # ------------------- Frame update: BOTH -------------------

    def _update_frame_both(self, frame_num):

        # Update done status: now "Done" if reached goal OR crashed
        done_cbf = (np.linalg.norm(self.p_cbf - self.goal) < 0.05) or self.cbf_crashed
        done_vf  = (np.linalg.norm(self.p_vf  - self.goal) < 0.05) or self.vf_crashed

        if done_cbf and done_vf:
            self._stop_simulation()
            # (Smoothness print logic stays here)
            return

        # --- CBF Agent Update ---
        R_stop = self.circ_R + 0.08  # R_eff (consistent with constraint margin)
        ms_cbf = 0.0
        u_cbf = np.zeros(2)
        if not done_cbf:
            dist_cbf = np.linalg.norm(self.p_cbf - self.circ_c)
            speed_cbf = np.linalg.norm(self.v_cbf)
            if dist_cbf <= R_stop or (dist_cbf <= R_stop * 1.08 and speed_cbf < 0.05):
                self.cbf_crashed = True
                print(f"CBF stopped at boundary. dist={dist_cbf:.3f}, speed={speed_cbf:.3f}")
            else:
                # Normal update logic only runs if NOT crashed
                u_cbf, _, _, ms_cbf = self._compute_control_step(self.p_cbf, self.v_cbf, self.u_prev_cbf, "cbf")
                self.u_prev_cbf = u_cbf.copy()
                p_next = self.p_cbf + self.dt * u_cbf
                p_next = self._repair_circle_penetration(p_next)
                self.v_cbf = (p_next - self.p_cbf) / (self.dt + EPS)
                self.p_cbf = p_next
                self.path_cbf.append(self.p_cbf.copy())

        # Step VFCBF agent
        ms_vf = 0.0
        u_vf = np.zeros(2)
        if not done_vf:
            dist_vf = np.linalg.norm(self.p_vf - self.circ_c)
            speed_vf = np.linalg.norm(self.v_vf)
            if dist_vf <= R_stop or (dist_vf <= R_stop * 1.08 and speed_vf < 0.05):
                self.vf_crashed = True
                print(f"VFCBF stopped at boundary. dist={dist_vf:.3f}, speed={speed_vf:.3f}")
            else:
                u_vf, _, _, ms_vf = self._compute_control_step(self.p_vf, self.v_vf, self.u_prev_vf, "vfcbf")
                self.u_prev_vf = u_vf.copy()
                p_next = self.p_vf + self.dt * u_vf
                p_next = self._repair_circle_penetration(p_next)
                self.v_vf = (p_next - self.p_vf) / (self.dt + EPS)
                self.p_vf = p_next
                self.path_vf.append(self.p_vf.copy())

        # Draw both trajectories
        pc = np.array(self.path_cbf)
        pv = np.array(self.path_vf)

        self.agent_cbf_artist.set_data([self.p_cbf[0]], [self.p_cbf[1]])
        self.trail_cbf_artist.set_data(pc[:, 0], pc[:, 1])

        self.agent_vf_artist.set_data([self.p_vf[0]], [self.p_vf[1]])
        self.trail_vf_artist.set_data(pv[:, 0], pv[:, 1])

        # Shared text block
        anchor = 0.5 * (self.p_cbf + self.p_vf)
        self.vfcbf_text.set_position((anchor[0] + 0.1, anchor[1] + 0.1))
        self.vfcbf_text.set_text(self._format_text_both(ms_cbf, ms_vf))

        # Control arrows for both
        if self.ctrl_quiv_cbf is None:
            self.ctrl_quiv_cbf = self.ax.quiver(
                [self.p_cbf[0]], [self.p_cbf[1]], [u_cbf[0]], [u_cbf[1]],
                angles='xy', scale_units='xy', scale=1.0,
                width=0.010, alpha=0.95, pivot='tail', color='m'
            )
            self.ax.quiverkey(self.ctrl_quiv_cbf, X=0.13, Y=1.06, U=1,
                              label='CBF control', labelpos='E')
        else:
            self.ctrl_quiv_cbf.set_offsets(np.array([[self.p_cbf[0], self.p_cbf[1]]]))
            self.ctrl_quiv_cbf.set_UVC([u_cbf[0]], [u_cbf[1]])

        if self.ctrl_quiv_vf is None:
            self.ctrl_quiv_vf = self.ax.quiver(
                [self.p_vf[0]], [self.p_vf[1]], [u_vf[0]], [u_vf[1]],
                angles='xy', scale_units='xy', scale=1.0,
                width=0.010, alpha=0.95, pivot='tail', color='c'
            )
            self.ax.quiverkey(self.ctrl_quiv_vf, X=0.13, Y=1.09, U=1,
                              label='VFCBF control', labelpos='E')
        else:
            self.ctrl_quiv_vf.set_offsets(np.array([[self.p_vf[0], self.p_vf[1]]]))
            self.ctrl_quiv_vf.set_UVC([u_vf[0]], [u_vf[1]])

        if frame_num >= self.steps - 1:
            self._stop_simulation()

        return (self.agent_cbf_artist, self.trail_cbf_artist,
                self.agent_vf_artist, self.trail_vf_artist,
                self.vfcbf_text)

    # --------- Non animated compare run (kept) ---------

    def run_open_loop_once(self, mode, steps=None):
        steps = steps or self.steps
        p = self.start.copy()
        g = self.goal.copy()
        v = np.zeros(2)
        u_prev = np.zeros(2)
        path = [p.copy()]
        for _ in range(steps):
            if np.linalg.norm(p - self.circ_c) <= self.circ_R:
                print(f"CRASH: {mode.upper()} hit obstacle.")
                break
            if np.linalg.norm(p - g) < 0.05:
                break
            u_nom = u_nominal(p, v, g, vmax=self.vmax)
            A, B = [], []
            circle_cons, R_eff, _ = circle_constraints_select(
                p, self.circ_c, self.circ_R,
                gamma=self.gamma_c, tau=self.tau_c, beta=self.beta_c,
                mode=mode
            )
            for a_c, b_c in circle_cons:
                A.append(a_c)
                B.append(b_c)

            # Braking distance CBF
            brake = braking_cbf_constraint(
                p, v, self.circ_c, R_eff,
                a_max=self.a_max, alpha_brake=self.alpha_brake
            )
            if brake is not None:
                A.append(brake[0])
                B.append(brake[1])

            strict = []
            if mode in ("cbf", "both") and circle_cons:
                strict = [circle_cons[0]]

            u_proj = project_with_speed_and_repair(
                u_nom, A, B, vmax=self.vmax, strict_normals=strict, passes=12, reits=4
            )
            u = smooth_control(u_proj, u_prev, alpha=self.alpha_u)
            u_prev = u.copy()

            p_next = p + self.dt * u
            p_next = self._repair_circle_penetration(p_next)

            v = (p_next - p) / (self.dt + EPS)
            p = p_next
            path.append(p.copy())
        return np.array(path)

    def on_compare(self, _event=None):
        # clear old trails/arrows
        self.trail_artist.set_data([], [])
        self.agent_artist.set_data([], [])
        self.trail_cbf_artist.set_data([], [])
        self.agent_cbf_artist.set_data([], [])
        self.trail_vf_artist.set_data([], [])
        self.agent_vf_artist.set_data([], [])
        self.ctrl_quiv = None
        self.ctrl_quiv_cbf = None
        self.ctrl_quiv_vf = None

        path_cbf = self.run_open_loop_once("cbf")
        path_vf  = self.run_open_loop_once("vfcbf")

        self.ax.plot(path_cbf[:, 0], path_cbf[:, 1], 'm-', lw=2.2, label="CBF only (open-loop)")
        self.ax.plot(path_vf[:, 0],  path_vf[:, 1],  'c-', lw=2.2, label="VFCBF only (open-loop)")
        self.ax.legend(loc="upper left", fontsize=8)

        cbf_k2, cbf_htv, cbf_L = _path_smoothness_metrics(path_cbf)
        vf_k2,  vf_htv,  vf_L  = _path_smoothness_metrics(path_vf)

        print("\n=== PATH SMOOTHNESS (open-loop) ===")
        print(f"CBF   : S_kappa2/L = {cbf_k2:.6f},  HTV/L = {cbf_htv:.6f},  L = {cbf_L:.3f} m")
        print(f"VFCBF : S_kappa2/L = {vf_k2:.6f},   HTV/L = {vf_htv:.6f},   L = {vf_L:.3f} m")

        def rank_tuple(k2, htv): return (k2, htv)
        winner = "CBF" if rank_tuple(cbf_k2, cbf_htv) < rank_tuple(vf_k2, vf_htv) else "VFCBF"
        print(f"=> Smoother by metrics: {winner}")

        txt = (
            "Open-loop Smoothness (lower = smoother)\n"
            f"CBF   : Sκ²/L={cbf_k2:.4f}, HTV/L={cbf_htv:.4f}, L={cbf_L:.2f}m\n"
            f"VFCBF : Sκ²/L={vf_k2:.4f}, HTV/L={vf_htv:.4f}, L={vf_L:.2f}m\n"
            f"Winner: {winner}"
        )
        self.ax.text(
            0.50, 0.02, txt, transform=self.ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.4')
        )

        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    np.random.seed(2)
    InteractiveSimDrag()