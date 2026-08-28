r"""
Agile avoidance sandbox: one nonholonomic car, three controllers.

Controllers
    cbf    baseline. By default the standard distance HOCBF and nothing
           else: the gap barrier $h = s$, lifted once for relative
           degree two, with linear class K margins. Its conservatism
           and lateness are properties of the standard form. A toggle
           (GUI checkbox or --brake) adds the classical braking
           distance row, which encodes input awareness at the price of
           speed blindness: the two ends of the baseline dial.
    apf    baseline. Goal attraction plus Gaussian repulsion, followed
           directly. Swerves, but certifies nothing.
    vcbf   ours. One Gaussian repulsive field used three ways: as a
           preloading swerve reference, as an energy cap certificate, and
           as a momentum governor that credits transverse speed.

Plant
    Unicycle with forward speed and heading. Inputs are longitudinal
    acceleration $a$ and yaw rate $\omega$, both bounded. Lateral motion
    is not an input: the only route to it is the Lie bracket of drift and
    steering, $[f, g_\omega]$, whose magnitude equals the speed $s$. A
    stopped car cannot sidestep; a fast car sidesteps through steering
    with gain equal to its speed.

Math, per circular obstacle with gap $s = \|p - c\| - R_{eff}$, outward
normal $\hat n$ and tangent $\hat t = R_{90}\hat n$:

    field    $h(p) = \kappa(s)\,\hat n$,  $\kappa(s) = \beta e^{-s^2/(2\sigma^2)}$
    energy   $E(p) = \tfrac12\|h\|^2 = \tfrac12\beta^2 e^{-s^2/\sigma^2}$

$E$ is monotone decreasing in the gap, so an energy cap is a clearance
floor: $\{E \le E_{cap}\} = \{s \ge s_{cap}\}$. Along the motion, with
$v_r = \hat n^\top\dot p$, $v_\perp = \hat t^\top\dot p$, $\mu = \hat n\cdot e$,

    $\dot E  = E'(s)\, v_r$
    $\ddot E = E'' v_r^2 \; + \; (E'/d)\, v_\perp^2 \; + \; E'\,(\mu\, a - v_\perp\, \omega)$

Two structural facts follow from $E' < 0$ and drive the whole design.
First, transverse speed lowers $\ddot E$ quadratically: momentum carried
across the obstacle line is relief, not hazard. Second, steering reaches
the certificate with gain $|E'|\,|v_\perp|$, proportional to the transverse
speed already present and exactly zero on a dead center approach. A
constraint therefore cannot start the swerve. Only the reference can, and
it must start early. Preloading is that necessity made explicit, and the
preload horizon is computed from the actuator bounds rather than tuned.

Equivalently, in the Lie bracket language of the accompanying notes: along
a reference flow $\dot p = V(p)$ the field rate splits as
$\tfrac{d}{dt}h \approx [V, h]$, and $\langle [V,h], h\rangle = \dot E$ is the
parallel component while $[V,h]_{\perp h}$ is the lateral redirection. The
controller bounds the parallel component (certificate and governor) and
spends the perpendicular one early (preload reference).

Run with no arguments for the interactive GUI. Headless entry points:
    --selftest        derivative and constraint row checks, quick rollouts
    --matrix          status grid over scenarios and launch speeds
    --brake           give the baseline the braking distance row
    --alat X          override the lateral acceleration cap
    --diagnose NAME   telemetry figure for one scenario, saved as PNG
    --sweep           vmax sweep on the sprint geometry, cbf vs vcbf

Statuses: goal, STALLED (speed collapsed), PINNED (settled onto the
margin shell slowly), BOUNDARY (hit the shell at speed, a certificate
failure), COLLISION (inside the physical circle), timeout.
"""

import argparse
import time

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from matplotlib.animation import FuncAnimation

EPS = 1e-12


# ---------------------- Smoothness helpers ----------------------

def _path_length(P):
    P = np.asarray(P, float)
    if len(P) < 2:
        return 0.0
    seg = np.diff(P, axis=0)
    return float(np.sum(np.linalg.norm(seg, axis=1)))


def _path_smoothness_metrics(P, ds_floor=0.0):
    """
    Returns (S_kappa2_per_m, S_HTV_per_m, L) for a polyline path P (Nx2).
    Segments shorter than ds_floor are dropped from the curvature sum,
    because discrete curvature is $|\\Delta\\theta| / \\Delta s$ and a nearly
    stationary stretch reports enormous curvature from heading noise.
    """
    P = np.asarray(P, float)
    if len(P) < 3:
        return 0.0, 0.0, _path_length(P)

    seg = np.diff(P, axis=0)
    theta = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    ds = np.linalg.norm(seg, axis=1)

    dtheta = theta[1:] - theta[:-1]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    ds_local = 0.5 * (ds[1:] + ds[:-1])
    keep = ds_local >= ds_floor
    ds_local = np.where(ds_local < 1e-12, 1e-12, ds_local)

    kappa = np.abs(dtheta) / ds_local
    L = max(float(np.sum(ds)), 1e-12)
    S_k2 = float(np.sum((kappa[keep] ** 2) * ds_local[keep])) / L
    S_htv = float(np.sum(np.abs(dtheta[keep]))) / L
    return S_k2, S_htv, L


# ---------------------- Small vector utilities ----------------------

def _clip_norm(u, umax):
    n = float(np.linalg.norm(u))
    if n > umax:
        return u * (umax / (n + EPS))
    return np.asarray(u, float).copy()


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _perp(v):
    """$R_{90} v$, rotation by ninety degrees counterclockwise."""
    return np.array([-v[1], v[0]])


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ====================== Gaussian obstacle field ======================

def field_profiles(s, sigma, beta):
    """
    Closed form radial profiles of the Gaussian field and its energy.

        $\\kappa(s) = \\beta e^{-s^2/(2\\sigma^2)}$
        $E(s)      = \\tfrac12 \\kappa^2 = \\tfrac12 \\beta^2 e^{-s^2/\\sigma^2}$
        $E'(s)     = -(2s/\\sigma^2)\\, E$
        $E''(s)    = 2E\\,(2 s^2/\\sigma^4 - 1/\\sigma^2)$

    $E' \\le 0$ for $s \\ge 0$: the energy decays monotonically with
    clearance, which is what lets a cap on $E$ act as a clearance floor.
    The gradient vanishes only at $s = 0$, which sits strictly inside the
    terminal margin shell and outside the certified set.

    Returns (kappa, E, dE, ddE).
    """
    s = float(s)
    sig2 = sigma * sigma
    kap = beta * np.exp(-s * s / (2.0 * sig2))
    E = 0.5 * kap * kap
    dE = -(2.0 * s / sig2) * E
    ddE = 2.0 * E * (2.0 * s * s / (sig2 * sig2) - 1.0 / sig2)
    return kap, E, dE, ddE


# ====================== Position barriers ======================

class PositionBarrier:
    """
    A scalar barrier $B(p)$ with the rate requirement
    $\\dot B \\ge -\\rho(p)$, $\\rho \\ge 0$ on the safe set. Carrying the
    gradient, the Hessian and the gradient of $\\rho$ lets one lifting
    routine impose the requirement at relative degree two on the car:
    with $\\psi = \\dot B + \\rho$, enforce $\\dot\\psi \\ge -k\\,\\psi$. The
    Hessian term $\\dot p^\\top \\nabla^2 B\\, \\dot p$ is what carries the
    transverse relief, so it is not optional.
    """

    __slots__ = ("B", "gradB", "hessB", "rho", "grad_rho", "name")

    def __init__(self, B, gradB, hessB, rho, grad_rho, name=""):
        self.B = float(B)
        self.gradB = np.asarray(gradB, float)
        self.hessB = np.asarray(hessB, float)
        self.rho = float(rho)
        self.grad_rho = np.asarray(grad_rho, float)
        self.name = name


def distance_barrier(p, c, R_eff, alpha):
    """Baseline geometric barrier $B = s$ with $\\rho = \\alpha s$."""
    rel = p - c
    d = float(np.linalg.norm(rel)) + EPS
    n = rel / d
    P = np.eye(2) - np.outer(n, n)
    s = d - R_eff
    return PositionBarrier(B=s, gradB=n, hessB=P / d,
                           rho=alpha * s, grad_rho=alpha * n, name="dist")


def energy_barrier(p, c, R_eff, sigma, beta, E_cap, k1):
    """
    Our certificate: $B = E_{cap} - E(p)$ with $\\rho = k_1 B$.

        $\\nabla B  = -E'(s)\\,\\hat n$        (outward, since $E' < 0$)
        $\\nabla^2 B = -\\big(E'' \\hat n\\hat n^\\top + (E'/d) P_\\perp\\big)$

    The transverse block contributes $-(E'/d)\\, v_\\perp^2 > 0$ to
    $\\ddot B$: transverse speed enlarges the feasible input set at equal
    clearance, quadratically. That is the momentum awareness, exact, and
    it comes from the geometry rather than from a heuristic bonus term.
    """
    rel = p - c
    d = float(np.linalg.norm(rel)) + EPS
    n = rel / d
    P = np.eye(2) - np.outer(n, n)
    s = d - R_eff
    _, E, dE, ddE = field_profiles(s, sigma, beta)
    B = E_cap - E
    gradB = -dE * n
    hessB = -(ddE * np.outer(n, n) + (dE / d) * P)
    return PositionBarrier(B=B, gradB=gradB, hessB=hessB,
                           rho=k1 * B, grad_rho=k1 * gradB, name="energy")


# ====================== Relative degree one rows ======================

def brake_row(p, e, s_speed, c, R_eff, a_brk, alpha_b, s_cap, a_max):
    """
    Optional baseline add on, off by default (toggle in the GUI, or
    pass --brake). The braking distance CBF

        $h_b = 2 a_{brk}(s - s_{cap}) - \\|v\\|^2 \\ge 0$

    is not about the ability to brake, which the plant always has. It is
    the certificate admitting the vehicle has inertia: its superlevel
    set is the set of states from which stopping is still possible under
    the input bound, which the plain distance HOCBF with linear class K
    margins does not encode. The cost of that foresight is speed
    blindness: the yaw rate does not appear, all speed is hazard, and
    the baseline turns conservative everywhere. The toggle exposes the
    dial: off is late at speed, on is conservative at all speeds, and no
    setting of the dial yields agility.
    Returns (a_vec, b, h_b) with a_vec in normalized car inputs.
    """
    rel = p - c
    d = float(np.linalg.norm(rel)) + EPS
    n = rel / d
    s = d - R_eff
    v_r = s_speed * float(n @ e)
    h_b = 2.0 * a_brk * (s - s_cap) - s_speed * s_speed
    a_vec = np.array([-2.0 * s_speed * a_max, 0.0])
    b = -alpha_b * h_b - 2.0 * a_brk * v_r
    return a_vec, b, h_b


def governor_h(p, e, s_speed, c, R_eff, a_brk, gamma, s_cap):
    """Scalar value of the credited governor barrier, for telemetry."""
    rel = p - c
    d = float(np.linalg.norm(rel)) + EPS
    n = rel / d
    t = _perp(n)
    s = d - R_eff
    v_r = s_speed * float(n @ e)
    v_p = s_speed * float(t @ e)
    v_rm = max(-v_r, 0.0)
    return 2.0 * a_brk * (s - s_cap) - v_rm * v_rm + gamma * v_p * v_p


def governor_row(p, e, s_speed, c, R_eff, a_brk, gamma, alpha_g, s_cap,
                 a_max, omega_max):
    """
    Our momentum governor, the credited replacement for the brake row:

        $h_g = 2 a_{brk}(s - s_{cap}) - v_{r-}^2 + \\gamma\\, v_\\perp^2$,
        $v_{r-} = \\max(-v_r, 0)$.

    Only inward radial speed is hazard, and transverse speed is credit.
    The braking demand vanishes once $v_\\perp^2 \\ge v_{r-}^2/\\gamma$, an
    approach angle threshold of $\\arctan(1/\\sqrt{\\gamma})$ from the
    radial line; beyond it a committed swerve is certified at any speed,
    including the circling limit. Derivative, using $\\dot s = v_r$,
    $\\dot v_r = \\mu a - v_\\perp \\omega + v_\\perp^2/d$ and
    $\\dot v_\\perp = (\\hat t\\cdot e)\\, a + \\mu s \\omega - v_r v_\\perp / d$:

        $\\dot h_g = 2 a_{brk} v_r + \\chi\\, 2 v_{r-} \\dot v_r
                    + 2\\gamma v_\\perp \\dot v_\\perp$,   $\\chi = [v_r < 0]$.

    Read the $\\omega$ coefficient: $\\chi 2 v_{r-}(-v_\\perp)
    + 2\\gamma v_\\perp \\mu s$. Every term carries a factor of $v_\\perp$,
    so on a dead center approach the governor, like every constraint
    here, has no steering authority and can only brake. The preload
    reference exists precisely because of this vanishing coefficient.
    Returns (a_vec, b, h_g) with a_vec in normalized car inputs.
    """
    rel = p - c
    d = float(np.linalg.norm(rel)) + EPS
    n = rel / d
    t = _perp(n)
    s = d - R_eff
    mu = float(n @ e)
    te = float(t @ e)
    v_r = s_speed * mu
    v_p = s_speed * te
    v_rm = max(-v_r, 0.0)
    chi = 1.0 if v_r < 0.0 else 0.0

    h_g = 2.0 * a_brk * (s - s_cap) - v_rm * v_rm + gamma * v_p * v_p
    coef_a = chi * 2.0 * v_rm * mu + 2.0 * gamma * v_p * te
    coef_w = chi * 2.0 * v_rm * (-v_p) + 2.0 * gamma * v_p * mu * s_speed
    const = (2.0 * a_brk * v_r
             + chi * 2.0 * v_rm * (v_p * v_p / d)
             - 2.0 * gamma * v_p * v_p * v_r / d)
    a_vec = np.array([coef_a * a_max, coef_w * omega_max])
    b = -alpha_g * h_g - const
    return a_vec, b, h_g


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
            if ak @ u - bk < 0.0:
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
            M = np.vstack([A_mat[i], A_mat[j]])
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
        dist = [np.linalg.norm(c - u_nom) for c in feas]
        return feas[int(np.argmin(dist))]

    viol = A_mat @ u_nom - b_vec
    k = int(np.argmin(viol))
    step = (b_vec[k] - A_mat[k] @ u_nom) / (A_mat[k] @ A_mat[k] + EPS)
    return u_nom + max(0.0, step) * A_mat[k]


def project_with_bounds_and_repair(u_nom, A, b, clamp, strict_normals=None,
                                   passes=12, reits=4):
    """
    Alternating projection between the constraint rows and the input box.
    The loop exits on the input set: input bounds are hard and barrier
    rows are not, so when the two cannot be satisfied together the row is
    genuinely infeasible at that state and max_violation is the honest
    way to see it, rather than an unbounded input.
    """
    u = project_qp_like(u_nom, A, b, passes=passes)
    for _ in range(reits):
        u = clamp(u)
        u = project_qp_like(u, A, b, passes=passes)
        if strict_normals:
            for (ai, bi) in strict_normals:
                if ai @ u - bi < 0.0:
                    step = (bi - ai @ u) / (ai @ ai + EPS)
                    u = u + step * ai
    return clamp(u)


def max_violation(A, b, u):
    if not A:
        return 0.0
    return float(max(bi - ai @ u for ai, bi in zip(A, b)))


# ====================== The car ======================

class Car:
    """
    Nonholonomic unicycle with a curvature limit at speed. State is
    position $p$, heading $\\theta$ and forward speed $s$:

        $\\dot p = s\\, e(\\theta)$, $\\dot\\theta = \\omega$, $\\dot s = a$,
        $|a| \\le a_{max}$, $|\\omega| \\le \\min(\\omega_{max},\\, a_{lat}/s)$.

    The second bound is the decoupled friction circle: lateral
    acceleration $s\\,\\omega$ may not exceed $a_{lat}$, so the turn radius
    is at least $s^2/a_{lat}$ at speed. This is what makes last minute
    turning physically impossible for a fast vehicle, for every
    controller equally. At low speed the geometric steering bound
    $\\omega_{max}$ takes over.

    World acceleration: $\\ddot p = a\\, e + s\\, \\omega\\, e_\\perp$, so a
    position barrier row reads

        $(\\nabla B\\cdot e)\\, a \\; + \\; s\\, (\\nabla B\\cdot e_\\perp)\\, \\omega \\;\\ge\\; b.$

    The yaw coefficient carries a factor of $s$: steering reaches any
    position certificate only in proportion to current speed, and not at
    all at rest. That is the bracket $[f, g_\\omega] = -s\\, e_\\perp$ in
    coordinates. Inputs are normalized to $(a/a_{max}, \\omega/\\omega_{max})$
    so the projection cost never compares an acceleration to a yaw rate;
    the friction circle tightens the normalized yaw box to
    $\\pm\\min(1, a_{lat}/(s\\,\\omega_{max}))$.
    """

    goal_tol = 0.18
    input_labels = ("a/amax", "w/wmax")

    def __init__(self, vmax=2.5, a_max=3.0, omega_max=2.5, s_min=0.0,
                 k_speed=3.2, k_head=3.0, a_lat=np.inf):
        self.vmax = vmax
        self.a_max = a_max
        self.omega_max = omega_max
        self.s_min = s_min
        self.k_speed = k_speed
        self.k_head = k_head
        self.a_lat = float(a_lat)
        self._p = np.zeros(2)
        self._th = 0.0
        self._s = 0.0

    def omega_frac_cap(self):
        """Normalized yaw bound from the friction circle at the current
        speed: $\\min(1, a_{lat}/(s\\,\\omega_{max}))$, with the geometric
        bound recovered below a small speed floor."""
        s_eff = max(self._s, 0.3)
        return float(min(1.0, self.a_lat / (s_eff * self.omega_max)))

    def reset(self, p0, theta0=0.0, speed0=0.0):
        self._p = np.array(p0, float)
        self._th = float(theta0)
        self._s = float(np.clip(speed0, self.s_min, self.vmax))

    @property
    def p(self):
        return self._p

    @property
    def heading(self):
        return np.array([np.cos(self._th), np.sin(self._th)])

    @property
    def left(self):
        return np.array([-np.sin(self._th), np.cos(self._th)])

    @property
    def theta(self):
        return self._th

    @property
    def pdot(self):
        return self._s * self.heading

    @property
    def speed(self):
        return self._s

    def denorm(self, u_hat):
        return np.array([u_hat[0] * self.a_max, u_hat[1] * self.omega_max])

    def barrier_row(self, bar, k_ho):
        """
        Lift a PositionBarrier to a row $a^\\top \\hat u \\ge b$ in normalized
        inputs, at relative degree two:

            $\\nabla B\\cdot \\ddot p \\ge -k(\\nabla B\\cdot v + \\rho)
              - v^\\top \\nabla^2 B\\, v - \\nabla\\rho\\cdot v$.
        """
        e = self.heading
        e_perp = self.left
        s = self._s
        pd = s * e
        a_vec = np.array([
            float(bar.gradB @ e) * self.a_max,
            s * float(bar.gradB @ e_perp) * self.omega_max,
        ])
        b = (-k_ho * (float(bar.gradB @ pd) + bar.rho)
             - float(pd @ (bar.hessB @ pd))
             - float(bar.grad_rho @ pd))
        return a_vec, b

    def clamp(self, u):
        u = np.asarray(u, float)
        wcap = self.omega_frac_cap()
        return np.array([float(np.clip(u[0], -1.0, 1.0)),
                         float(np.clip(u[1], -wcap, wcap))])

    def track_velocity(self, v_des, commit=0.0):
        """
        Turn a desired world velocity into normalized inputs. The speed
        reference is gated by $\\max(\\cos(\\text{err}), \\text{commit})$: with
        commit at zero the car slows fully while turning (the baselines);
        a positive floor keeps forward commitment through the swerve,
        which is part of the committed passage behavior of ours.
        """
        v_ref = _clip_norm(v_des, self.vmax)
        speed_des = float(np.linalg.norm(v_ref))
        if speed_des < 1e-6:
            return self.clamp(np.array([-self.k_speed * self._s / self.a_max,
                                        0.0]))
        th_d = float(np.arctan2(v_ref[1], v_ref[0]))
        err = _wrap(th_d - self._th)
        omega = self.k_head * err
        s_ref = speed_des * max(np.cos(err), commit)
        a = self.k_speed * (s_ref - self._s)
        return self.clamp(np.array([a / self.a_max, omega / self.omega_max]))

    def step(self, dt, u_hat, repair):
        a, omega = self.denorm(u_hat)
        self._s = float(np.clip(self._s + dt * a, self.s_min, self.vmax))
        self._th = _wrap(self._th + dt * omega)
        p_prev = self._p.copy()
        self._p = repair(p_prev + dt * self._s * self.heading)

    def kill_inward(self, n_hat):
        if float(self.heading @ n_hat) * self._s < 0.0:
            self._s = 0.0


# ====================== References ======================

def goal_field(p, goal, k_att, vmax):
    return _clip_norm(k_att * (goal - p), vmax)


def apf_reference(p, goal, obstacles, prm):
    """
    Classic potential field with a Gaussian repulsion, followed directly.
    Attractive pull is conic beyond d_switch so it does not blow up at
    range. No barrier, no constraint, no projection: the point of the
    baseline is that an APF already swerves, it simply certifies nothing
    and can be outrun by a plant that cannot track the field it is fed.
    """
    to_goal = goal - p
    dg = float(np.linalg.norm(to_goal)) + EPS
    if dg > prm["apf_dswitch"]:
        F = prm["apf_zeta"] * prm["apf_dswitch"] * to_goal / dg
    else:
        F = prm["apf_zeta"] * to_goal

    rep = np.zeros(2)
    for (c, R) in obstacles:
        rel = p - c
        d = float(np.linalg.norm(rel)) + EPS
        n = rel / d
        s = max(d - (R + prm["margin"]), 0.0)
        rep = rep + prm["apf_eta"] * np.exp(-s * s /
                                            (2.0 * prm["apf_sig"] ** 2)) * n
    return _clip_norm(F + rep, prm["vmax"])


def preload_horizon(speed, prm):
    """
    The range at which the preload gate opens, computed from the bounds.
    The available yaw rate at speed is the friction limited

        $\\omega_{eff}(v) = \\min(\\omega_{max},\\; a_{lat}/v)$,

    and the horizon is the distance covered while yawing through the
    planned swerve angle at that rate, plus a fraction of the stopping
    distance, plus a floor:

        $D_{pre}(v) = v\\, \\Delta\\theta^* / \\omega_{eff}(v)
                     + \\xi\\, v^2 / (2 a_{max}) + D_0.$

    Past the friction knee the first term grows as
    $v^2 \\Delta\\theta^* / a_{lat}$: a fast car must preload
    quadratically earlier, exactly because it cannot turn late. This
    replaces the usual hand tuned influence radius with a quantity the
    actuator limits dictate.
    """
    w_eff = min(prm["omega_max"], prm["a_lat"] / max(speed, 0.3))
    return (speed * prm["dtheta_pre"] / w_eff
            + prm["xi_pre"] * speed * speed / (2.0 * prm["a_max"])
            + prm["d0_pre"])


def vcbf_reference(p, speed, goal, obstacles, prm, latch=None):
    """
    The preloaded swerve reference: goal field, plus a radial term from
    the field itself, plus a tangential term along $\\pm\\hat t$, which is
    the direction of the perpendicular bracket component $[V, h]_{\\perp h}$
    for this radial field. The tangential gate opens at the computed
    preload horizon, and the side is chosen once per obstacle (goalward
    tie break, latched) because at the exactly symmetric configuration
    the bracket component vanishes and something must break the tie.

    Frontness is a corridor test, not a heading proxy: an obstacle is
    in front only while its center sits ahead along the sight line to
    the goal, short of the goal, and within one lane of that line. A
    dot product proxy such as $\\hat n \\cdot \\hat g$ stays negative for the
    whole circumnavigation and turns the composite into a circulating
    field with a limit cycle at speed. The corridor test switches the
    tangential term off the moment the sight line clears the disk.

    Returns the reference velocity and an engagement level in $[0, 1]$,
    the largest product of frontness and preload gate over obstacles.
    Engagement gates the commit floor in tracking: commitment is for
    the middle of a swerve, not for the final approach to the goal.
    """
    if latch is None:
        latch = {}
    to_goal = goal - p
    dg = float(np.linalg.norm(to_goal)) + EPS
    ghat = to_goal / dg
    v = goal_field(p, goal, prm["k_att"], prm["vmax"])
    Dpre = preload_horizon(speed, prm)
    engage = 0.0

    for i, (c, R) in enumerate(obstacles):
        rel = p - c
        d = float(np.linalg.norm(rel)) + EPS
        n = rel / d
        t = _perp(n)
        s = d - (R + prm["margin"])
        oc = c - p
        a_los = float(ghat @ oc)                     # along the sight line
        b_los = float(np.linalg.norm(oc - a_los * ghat))   # off the line
        lane = R + prm["margin"] + prm["lane_pre"]
        front = (_smoothstep(a_los / 0.3)
                 * _smoothstep((dg + R - a_los) / 0.3)
                 * _smoothstep((lane - b_los) / 0.3))
        w = _smoothstep((Dpre - s) / prm["w_pre"])
        proj = float(t @ to_goal)
        tie = 1.0 if abs(proj) < 1e-9 else float(np.sign(proj))
        # Latch the side while the corridor is blocked, release once the
        # sight line is clear. Hysteresis keeps the choice stable while
        # the instantaneous tie break flips during the swerve.
        if i in latch and front < 0.05:
            del latch[i]
        if i not in latch and w > 0.35 and front > 0.5:
            latch[i] = tie
        side = latch.get(i, tie)

        kr = np.exp(-max(s, 0.0) ** 2 / (2.0 * prm["sigma_r"] ** 2))
        v = v + prm["c_r"] * prm["vmax"] * kr * (0.1 + 0.9 * front) * n
        v = v + prm["c_t"] * prm["vmax"] * w * front * side * t
        engage = max(engage, front * w)

    # Strip the inward radial component of the composite very close to
    # each obstacle. The certificate would veto that component anyway,
    # and a reference that demands a vetoed direction parks the car at
    # the floor with the throttle clamped to zero. Rotating the demand
    # to the tangent instead lets the car pivot out and keep moving.
    for (c, R) in obstacles:
        rel = p - c
        d = float(np.linalg.norm(rel)) + EPS
        n = rel / d
        s = d - (R + prm["margin"])
        blend = np.exp(-max(s, 0.0) ** 2 / (2.0 * 0.35 ** 2))
        inward = min(0.0, float(v @ n))
        v = v - blend * inward * n
    return _clip_norm(v, prm["vmax"]), engage


# ====================== Agents ======================

CONTROLLER_SPECS = (
    ("cbf",  "CBF",  "m",      "-"),
    ("apf",  "APF",  "orange", "--"),
    ("vcbf", "VCBF", "c",      "-"),
)


class Agent:
    """One controller instance with its own car, path and diagnostics."""

    def __init__(self, key, label, color, linestyle):
        self.key = key
        self.label = label
        self.color = color
        self.linestyle = linestyle
        self.plant = None
        self.path = []
        self.done = False
        self.status = "idle"
        self.compute_ms = 0.0
        self.u_prev = np.zeros(2)
        self.last_u = np.zeros(2)
        self.min_clearance = np.inf
        self.infeas_steps = 0
        self.steps_taken = 0
        self.stall_count = 0
        self.contact_speed = 0.0
        self.telemetry = []
        self.state = {}
        self.line = None
        self.dot = None
        self.quiv = None

    def reset(self, plant, p0, theta0, speed0=0.0):
        self.plant = plant
        plant.reset(p0, theta0, speed0)
        self.path = [plant.p.copy()]
        self.done = False
        self.status = "running"
        self.compute_ms = 0.0
        self.u_prev = np.zeros(2)
        self.last_u = np.zeros(2)
        self.min_clearance = np.inf
        self.infeas_steps = 0
        self.steps_taken = 0
        self.stall_count = 0
        self.contact_speed = 0.0
        self.telemetry = []
        self.state = {}


# ====================== Scenarios ======================

def _S(start, goal, obstacles, vmax, v0_frac, blurb):
    return dict(start=np.array(start, float), goal=np.array(goal, float),
                obstacles=[(np.array(c, float), float(r))
                           for c, r in obstacles],
                vmax=float(vmax), v0_frac=float(v0_frac), blurb=blurb)


SCENARIOS = {
    "headon": _S((-2.0, -1.1), (2.0, 1.2), [((0.2, 0.0), 0.85)], 2.5, 0.6,
                 "single obstacle, near head on approach at moderate speed"),
    "wall":   _S((-2.4, 0.0), (2.3, 0.0), [((0.2, 0.0), 1.25)], 2.5, 0.6,
                 "goal directly behind a wide obstacle, the classic"
                 " potential field local minimum geometry, exactly"
                 " symmetric"),
    "gap":    _S((-2.2, 0.0), (2.3, 0.0),
                 [((0.2, 0.9), 0.6), ((0.2, -0.9), 0.6)], 2.5, 0.6,
                 "narrow corridor between two obstacles"),
    "trap":   _S((-2.2, 0.8), (2.3, 0.8),
                 [((0.2, 0.9), 0.6), ((0.2, -0.9), 0.6)], 2.5, 0.6,
                 "the composition: the upper obstacle sits dead ahead and"
                 " the only way through is a committed swerve into the"
                 " corridor"),
    "sprint": _S((-2.3, 0.0), (2.4, 0.0), [((0.4, 0.0), 0.6)], 4.0, 1.0,
                 "high speed launch straight at the obstacle. The braking"
                 " distance exceeds the available run in, so a certificate"
                 " that can only brake is too late by construction"),
}


# ====================== Interactive Sim ======================

class InteractiveSimDrag:
    def __init__(self, show=True, scenario=None, v0_frac=None, vmax=None,
                 brake=False, a_lat_override=None):
        # Arena limits
        self.xmin, self.xmax = -2.5, 2.7
        self.ymin, self.ymax = -2.5, 2.5

        # Obstacles, list of (centre, radius)
        self.obstacles = [(np.array([0.2, 0.0]), 0.85)]
        self.margin = 0.06
        self.scenario = "headon"

        # Shared field parameters
        self.sigma = 0.55
        self.beta = 1.0
        self.s_cap = 0.12          # certified clearance floor, above margin
        # $E_{cap} = E(s_{cap})$ computed on demand via _E_cap()

        # Ours: certificate and governor
        self.k1_E = 3.0            # $\rho = k_1 B$ inside the energy HOCBF
        self.k2_E = 3.0            # outer gain of the lifting
        self.gamma_perp = 2.5      # transverse credit in the governor
        self.alpha_gov = 2.5

        # Ours: preload reference
        self.dtheta_pre = 1.3      # planned swerve angle $\Delta\theta^*$
        self.xi_pre = 0.5
        self.d0_pre = 0.35
        self.w_pre = 0.4
        self.lane_pre = 0.35       # corridor half width beyond $R$ + margin
        self.c_t = 1.0
        self.c_r = 0.6
        self.sigma_r = 0.7
        self.commit = 0.35

        # Baseline CBF
        self.alpha_dist = 2.0
        self.k_hocbf = 3.0
        self.brake_frac = 0.7      # $a_{brk} = 0.7\, a_{max}$
        self.alpha_brake = 1.5     # class K gain of the optional brake row
        self.cbf_brake = bool(brake)  # toggle: baseline brake distance CBF

        # APF baseline
        self.apf_zeta = 1.6
        self.apf_eta = 2.2
        self.apf_sig = 0.6
        self.apf_dswitch = 1.0

        # Car input bounds
        self.a_max = 3.0
        self.omega_max = 2.5
        self.a_lat = 4.0           # lateral cap: turn radius $\ge v^2/a_{lat}$
        if a_lat_override is not None:
            self.a_lat = float(a_lat_override)

        # Sim
        self.dt = 0.028
        self.steps = 1800
        self.vmax = 2.5
        self.v0_frac = 0.6
        self.k_att = 1.6
        self.activate_dist = 2.6   # fixed reach of the position barriers

        # Vector field grid
        self.grid_Nx = 30
        self.grid_Ny = 26

        self.goal = np.array([2.0, 1.2])
        self.start = np.array([-2.0, -1.1])

        self.mode = "vcbf"         # field view selection

        if scenario is not None:
            self.apply_scenario(scenario)
        if vmax is not None:
            self.vmax = float(vmax)
        if v0_frac is not None:
            self.v0_frac = float(np.clip(v0_frac, 0.0, 1.0))

        self.anim = None
        self.running = False
        self.dragging_artist = None

        self.agents = {key: Agent(key, label, color, ls)
                       for key, label, color, ls in CONTROLLER_SPECS}
        self.active = []

        if not show:
            return

        self.fig, self.ax = plt.subplots(figsize=(12.5, 8.5))
        plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.27)

        self._build_widgets()
        self._draw_static_scene()
        self._make_artists()
        self.on_reset(None)

        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        plt.show()

    # ------------------- parameter bundles -------------------

    def _E_cap(self):
        _, E, _, _ = field_profiles(self.s_cap, self.sigma, self.beta)
        return E

    def a_brk(self):
        return self.brake_frac * self.a_max

    def prm(self):
        return {
            "vmax": self.vmax, "margin": self.margin, "k_att": self.k_att,
            "a_max": self.a_max, "omega_max": self.omega_max,
            "a_lat": self.a_lat,
            "dtheta_pre": self.dtheta_pre, "xi_pre": self.xi_pre,
            "d0_pre": self.d0_pre, "w_pre": self.w_pre,
            "lane_pre": self.lane_pre,
            "c_t": self.c_t, "c_r": self.c_r, "sigma_r": self.sigma_r,
            "apf_zeta": self.apf_zeta, "apf_eta": self.apf_eta,
            "apf_sig": self.apf_sig, "apf_dswitch": self.apf_dswitch,
        }

    def make_plant(self):
        return Car(vmax=self.vmax, a_max=self.a_max,
                   omega_max=self.omega_max, a_lat=self.a_lat)

    # ------------------- obstacles and scenarios -------------------

    @property
    def ds_floor(self):
        return 0.1 * self.vmax * self.dt

    def nearest_obstacle(self, p):
        best = None
        for (c, R) in self.obstacles:
            d = float(np.linalg.norm(p - c)) + EPS
            if best is None or d - R < best[2] - best[1]:
                best = (c, R, d, (p - c) / d)
        return best

    def _inside_any(self, q):
        for (c, R) in self.obstacles:
            if float(np.linalg.norm(q - c)) < R + self.margin - 1e-6:
                return True
        return False

    def apply_scenario(self, key):
        sc = SCENARIOS[key]
        self.scenario = key
        self.start = sc["start"].copy()
        self.goal = sc["goal"].copy()
        self.obstacles = [(c.copy(), r) for c, r in sc["obstacles"]]
        self.vmax = sc["vmax"]
        self.v0_frac = sc["v0_frac"]

    # ------------------- widgets -------------------

    def _build_widgets(self):
        w, y, hgt = 0.128, 0.025, 0.05
        xs = [0.020 + i * 0.140 for i in range(7)]

        ax_cbf = self.fig.add_axes([xs[0], y, w, hgt])
        ax_apf = self.fig.add_axes([xs[1], y, w, hgt])
        ax_vf = self.fig.add_axes([xs[2], y, w, hgt])
        ax_all = self.fig.add_axes([xs[3], y, w, hgt])
        ax_dgn = self.fig.add_axes([xs[4], y, w, hgt])
        ax_cmp = self.fig.add_axes([xs[5], y, w, hgt])
        ax_rst = self.fig.add_axes([xs[6], y, w, hgt])

        self.btn_cbf = Button(ax_cbf, "Run CBF")
        self.btn_apf = Button(ax_apf, "Run APF")
        self.btn_vf = Button(ax_vf, "Run VCBF")
        self.btn_all = Button(ax_all, "Run All")
        self.btn_dgn = Button(ax_dgn, "Diagnose")
        self.btn_cmp = Button(ax_cmp, "Compare")
        self.btn_rst = Button(ax_rst, "Reset")

        self.btn_cbf.on_clicked(lambda e: self._start_run(["cbf"]))
        self.btn_apf.on_clicked(lambda e: self._start_run(["apf"]))
        self.btn_vf.on_clicked(lambda e: self._start_run(["vcbf"]))
        self.btn_all.on_clicked(
            lambda e: self._start_run(["cbf", "apf", "vcbf"]))
        self.btn_dgn.on_clicked(self.on_diagnose)
        self.btn_cmp.on_clicked(self.on_compare)
        self.btn_rst.on_clicked(self.on_reset)

        ax_mode = self.fig.add_axes([0.030, 0.095, 0.105, 0.095])
        ax_scn = self.fig.add_axes([0.150, 0.095, 0.150, 0.095])

        self.rad_mode = RadioButtons(ax_mode, ("cbf", "apf", "vcbf"),
                                     active=2)
        keys = tuple(SCENARIOS)
        self.rad_scn = RadioButtons(ax_scn, keys,
                                    active=keys.index(self.scenario))
        self.rad_mode.on_clicked(self.on_mode_change)
        self.rad_scn.on_clicked(self.on_scenario_change)

        self.fig.text(0.030, 0.196, "Field view", fontsize=9, weight="bold")
        self.fig.text(0.150, 0.196, "Scenario", fontsize=9, weight="bold")

        ax_brk = self.fig.add_axes([0.330, 0.100, 0.150, 0.075])
        ax_brk.set_frame_on(False)
        self.chk_brake = CheckButtons(ax_brk, ["CBF brake row"],
                                      [self.cbf_brake])
        self.chk_brake.on_clicked(self.on_brake_toggle)
        self.fig.text(0.330, 0.196, "Baseline option", fontsize=9,
                      weight="bold")

        ax_v0 = self.fig.add_axes([0.510, 0.125, 0.230, 0.030])
        self.sld_v0 = Slider(ax_v0, "v0 / vmax  ", 0.0, 1.0,
                             valinit=self.v0_frac, valstep=0.05)
        self.sld_v0.on_changed(self.on_v0)
        self.fig.text(0.510, 0.170, "initial speed", fontsize=9,
                      weight="bold")

    def _make_artists(self):
        (self.goal_artist,) = self.ax.plot(
            [self.goal[0]], [self.goal[1]], "rx", ms=12, mew=2.2,
            label="Goal", picker=8)
        (self.start_artist,) = self.ax.plot(
            [self.start[0]], [self.start[1]], "g^", ms=10, label="Start",
            picker=8)

        for ag in self.agents.values():
            (ag.line,) = self.ax.plot([], [], color=ag.color,
                                      ls=ag.linestyle, lw=2.0, alpha=0.9,
                                      label=f"{ag.label} traj")
            (ag.dot,) = self.ax.plot([], [], "o", color=ag.color, ms=8,
                                     label=f"{ag.label} agent")
            ag.quiv = None

        self.start_coord_text = self.ax.text(
            self.start[0], self.start[1] + 0.15,
            f"({self.start[0]:.2f}, {self.start[1]:.2f})", fontsize=9,
            ha="center")
        self.goal_coord_text = self.ax.text(
            self.goal[0], self.goal[1] - 0.2,
            f"({self.goal[0]:.2f}, {self.goal[1]:.2f})", fontsize=9,
            ha="center")

        self.info_text = self.ax.text(
            0.985, 0.02, "", transform=self.ax.transAxes, fontsize=8,
            family="monospace", ha="right", va="bottom",
            bbox=dict(facecolor="white", alpha=0.85,
                      boxstyle="round,pad=0.35"))

        self.quiv = None
        self._draw_or_update_quiver()
        self.ax.legend(loc="upper left", fontsize=7, ncol=2)

    # ------------------- scene -------------------

    def _ring_radius(self, R):
        """Mode specific overlay ring: for cbf the head on bite radius
        of the standard HOCBF at the slider speed, $s^* = v_0/\\alpha_1$
        (where $\\psi = \\dot h + \\alpha_1 h$ first hits zero on a straight
        approach), for vcbf the computed preload horizon, for apf the
        reach of the Gaussian repulsion."""
        v0 = self._initial_speed()
        if self.mode == "cbf":
            return R + self.margin + v0 / max(self.alpha_dist, EPS)
        if self.mode == "vcbf":
            return R + self.margin + preload_horizon(v0, self.prm())
        return R + self.margin + 2.0 * self.apf_sig

    def _mode_color(self):
        return {k: col for k, _l, col, _ls in CONTROLLER_SPECS}[self.mode]

    def _draw_static_scene(self):
        self.ax.clear()
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_aspect("equal", "box")
        self.ax.set_title(self._title_text(), fontsize=10)
        self.mode_rings = []
        for (c, R) in self.obstacles:
            self.ax.add_patch(plt.Circle(tuple(c), R, color="k",
                                         fill=False, lw=2))
            self.ax.add_patch(plt.Circle(tuple(c), R + self.margin,
                                         color="k", fill=False, lw=0.8,
                                         ls=":", alpha=0.6))
            ring = plt.Circle(tuple(c), self._ring_radius(R),
                              color=self._mode_color(), fill=False,
                              lw=1.0, ls="--", alpha=0.55)
            self.ax.add_patch(ring)
            self.mode_rings.append(ring)

    def _update_rings(self):
        for ring, (c, R) in zip(self.mode_rings, self.obstacles):
            ring.set_radius(self._ring_radius(R))
            ring.set_color(self._mode_color())

    def _title_text(self):
        v0 = self._initial_speed()
        r_turn = max(v0 / max(self.omega_max, EPS),
                     v0 * v0 / max(self.a_lat, EPS))
        ring_name = {"cbf": "HOCBF bite radius",
                     "apf": "field reach",
                     "vcbf": "preload horizon"}[self.mode]
        return (f"scenario: {self.scenario}   field view: "
                f"{self.mode.upper()}   vmax {self.vmax:.1f}   "
                f"v0 {v0:.2f} m/s   turn radius at v0 {r_turn:.2f} m   "
                f"a_lat {self.a_lat:.1f}   brake row "
                f"{'on' if self.cbf_brake else 'off'}\n"
                f"drag goal and start; quiver is the {self.mode.upper()} "
                f"reference at the slider speed; dashed ring: {ring_name}")

    def _redraw_scene_and_readd_artists(self):
        self._draw_static_scene()
        self._make_artists()

    # --------- Vector field view ----------

    def _field_vector_at(self, q):
        if self.mode == "apf":
            return apf_reference(q, self.goal, self.obstacles, self.prm())
        if self.mode == "vcbf":
            return vcbf_reference(q, self._initial_speed(), self.goal,
                                  self.obstacles, self.prm())[0]
        return goal_field(q, self.goal, self.k_att, self.vmax)

    def _compute_vector_field(self):
        xs = np.linspace(self.xmin, self.xmax, self.grid_Nx)
        ys = np.linspace(self.ymin, self.ymax, self.grid_Ny)
        XX, YY = np.meshgrid(xs, ys)
        Ux = np.zeros_like(XX)
        Uy = np.zeros_like(YY)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                q = np.array([XX[i, j], YY[i, j]])
                if self._inside_any(q):
                    continue
                u = self._field_vector_at(q)
                Ux[i, j], Uy[i, j] = u[0], u[1]

        M = np.hypot(Ux, Uy)
        M_safe = M + 1e-9
        rel = np.clip(M / (self.vmax + 1e-9), 0.2, 1.0)
        Ux_vis = (Ux / M_safe) * rel
        Uy_vis = (Uy / M_safe) * rel
        for (c, R) in self.obstacles:
            mask = (np.hypot(XX - c[0], YY - c[1]) < R + self.margin - 1e-6)
            Ux_vis[mask] = 0.0
            Uy_vis[mask] = 0.0
        return XX, YY, Ux_vis, Uy_vis

    def _draw_or_update_quiver(self):
        XX, YY, Ux, Uy = self._compute_vector_field()
        if self.quiv is None:
            self.quiv = self.ax.quiver(
                XX, YY, Ux, Uy, angles="xy", scale_units="xy", scale=18,
                width=0.006, alpha=0.8, pivot="mid", minlength=0)
        else:
            self.quiv.set_UVC(Ux, Uy)

    # ------------------- radio and slider callbacks -------------------

    def on_mode_change(self, label):
        self.mode = label
        self._update_rings()
        self._draw_or_update_quiver()
        self.ax.set_title(self._title_text(), fontsize=10)
        self.fig.canvas.draw_idle()

    def on_scenario_change(self, label):
        self.apply_scenario(label)
        if hasattr(self, "sld_v0"):
            self.sld_v0.set_val(self.v0_frac)
        self.on_reset(None)

    def on_brake_toggle(self, _label):
        self.cbf_brake = not self.cbf_brake
        self.ax.set_title(self._title_text(), fontsize=10)
        self.fig.canvas.draw_idle()

    def on_v0(self, val):
        self.v0_frac = float(val)
        if hasattr(self, "ax"):
            self._update_rings()
            if self.mode == "vcbf":
                self._draw_or_update_quiver()
            self.ax.set_title(self._title_text(), fontsize=10)
            self.fig.canvas.draw_idle()

    def on_diagnose(self, _event=None):
        if self.running:
            return
        self.diagnose_current(show=True)

    # ------------------- dragging -------------------

    def on_pick(self, event):
        if self.running:
            return
        if event.artist in (self.goal_artist, self.start_artist):
            self.dragging_artist = event.artist

    def on_motion(self, event):
        if (self.running or self.dragging_artist is None
                or event.inaxes != self.ax):
            return
        x = float(np.clip(event.xdata, self.xmin, self.xmax))
        y = float(np.clip(event.ydata, self.ymin, self.ymax))
        self.dragging_artist.set_data([x], [y])
        coord = f"({x:.2f}, {y:.2f})"
        if self.dragging_artist is self.goal_artist:
            self.goal[:] = [x, y]
            self.goal_coord_text.set_position((x, y - 0.2))
            self.goal_coord_text.set_text(coord)
        else:
            self.start[:] = [x, y]
            self.start_coord_text.set_position((x, y + 0.15))
            self.start_coord_text.set_text(coord)
        self._draw_or_update_quiver()
        self.fig.canvas.draw_idle()

    def on_release(self, _event):
        self.dragging_artist = None

    # ------------------- reset -------------------

    def _clear_agent_artists(self):
        for ag in self.agents.values():
            if ag.line is not None:
                ag.line.set_data([], [])
            if ag.dot is not None:
                ag.dot.set_data([], [])
            if ag.quiv is not None:
                ag.quiv.remove()
                ag.quiv = None
            ag.path = []
            ag.status = "idle"

    def on_reset(self, _event=None):
        if self.running and self.anim is not None:
            self.anim.event_source.stop()
        self.running = False
        self.anim = None
        self.active = []
        self._set_button_labels(running=False)
        self._redraw_scene_and_readd_artists()
        self._clear_agent_artists()
        self.info_text.set_text("")
        self.fig.canvas.draw_idle()

    def _set_button_labels(self, running):
        if not hasattr(self, "btn_cbf"):
            return
        self.btn_cbf.label.set_text("Running..." if running else "Run CBF")
        self.btn_apf.label.set_text("Running..." if running else "Run APF")
        self.btn_vf.label.set_text("Running..." if running else "Run VCBF")
        self.btn_all.label.set_text("Running..." if running else "Run All")

    # ------------------- geometry helpers -------------------

    def _initial_heading(self):
        d = self.goal - self.start
        return float(np.arctan2(d[1], d[0]))

    def _initial_speed(self):
        return self.v0_frac * self.vmax

    def _repair_circle_penetration(self, p_next):
        for _ in range(2):
            moved = False
            for (c, R) in self.obstacles:
                R_eff = R + self.margin
                rel = p_next - c
                d = float(np.linalg.norm(rel)) + EPS
                if d < R_eff:
                    p_next = c + (rel / d) * R_eff
                    moved = True
            if not moved:
                break
        return p_next

    def _act_dyn(self, speed):
        """Reach of the governor row, scaled with the braking distance
        at the current speed so the credit engages early enough to
        matter at any launch speed."""
        return max(self.activate_dist,
                   speed * speed / (2.0 * self.a_brk()) + 0.4)

    # ------------------- control step -------------------

    def _compute_control_step(self, agent):
        """One control step for one agent.
        Returns (u_applied, u_projected, viol, info)."""
        t0 = time.perf_counter()
        plant = agent.plant
        p = plant.p
        e = plant.heading
        speed = plant.speed
        E_cap = self._E_cap()
        a_brk = self.a_brk()
        act_dyn = self._act_dyn(speed)

        if agent.key == "apf":
            v_des = apf_reference(p, self.goal, self.obstacles, self.prm())
            u = plant.clamp(plant.track_velocity(v_des, commit=0.0))
            info = {"u_nom": u.copy(), "n_rows": 0}
            agent.compute_ms = (time.perf_counter() - t0) * 1000.0
            agent.u_prev = u.copy()
            return u, u, 0.0, info

        if agent.key == "cbf":
            u_nom = plant.track_velocity(
                goal_field(p, self.goal, self.k_att, self.vmax), commit=0.0)
            A, B, strict = [], [], []
            for (c, R) in self.obstacles:
                d = float(np.linalg.norm(p - c))
                s = d - (R + self.margin)
                if s <= self.activate_dist:
                    bar = distance_barrier(p, c, R + self.margin,
                                           self.alpha_dist)
                    row = plant.barrier_row(bar, self.k_hocbf)
                    A.append(row[0])
                    B.append(row[1])
                    strict.append(row)
                if self.cbf_brake and s <= act_dyn:
                    av, bv, _ = brake_row(p, e, speed, c, R + self.margin,
                                          a_brk, self.alpha_brake,
                                          self.s_cap, self.a_max)
                    A.append(av)
                    B.append(bv)
        else:  # vcbf, ours
            latch = agent.state.setdefault("side", {})
            v_des, engage = vcbf_reference(p, speed, self.goal,
                                           self.obstacles, self.prm(),
                                           latch=latch)
            u_nom = plant.track_velocity(
                v_des, commit=self.commit * _smoothstep(engage / 0.3))
            A, B, strict = [], [], []
            for (c, R) in self.obstacles:
                d = float(np.linalg.norm(p - c))
                s = d - (R + self.margin)
                if s <= self.activate_dist:
                    bar = energy_barrier(p, c, R + self.margin, self.sigma,
                                         self.beta, E_cap, self.k1_E)
                    row = plant.barrier_row(bar, self.k2_E)
                    A.append(row[0])
                    B.append(row[1])
                    strict.append(row)
                if s <= act_dyn:
                    av, bv, _ = governor_row(
                        p, e, speed, c, R + self.margin, a_brk,
                        self.gamma_perp, self.alpha_gov, self.s_cap,
                        self.a_max, self.omega_max)
                    A.append(av)
                    B.append(bv)

        u_proj = project_with_bounds_and_repair(
            u_nom, A, B, plant.clamp, strict_normals=strict,
            passes=14, reits=5)
        viol = max_violation(A, B, u_proj)
        info = {"u_nom": np.asarray(u_nom, float).copy(), "n_rows": len(A)}
        agent.compute_ms = (time.perf_counter() - t0) * 1000.0
        agent.u_prev = np.asarray(u_proj, float).copy()
        return u_proj, u_proj, viol, info

    def _advance_agent(self, agent):
        if agent.done:
            return
        plant = agent.plant
        c_near, R_near, d, n_hat = self.nearest_obstacle(plant.p)
        clearance = d - R_near
        agent.min_clearance = min(agent.min_clearance, clearance)

        R_stop = R_near + self.margin
        speed = plant.speed

        # Three outcomes at the boundary, kept distinct on purpose. The
        # repair clamps position onto the margin shell, so the physical
        # circle is essentially unreachable and what matters is the speed
        # at which the shell is touched: fast contact is a certificate
        # failure, settling onto it is the barrier driving $B$ to zero.
        if d <= R_near:
            agent.done = True
            agent.status = "COLLISION"
            agent.contact_speed = speed
            return
        if d <= R_stop:
            agent.done = True
            agent.contact_speed = speed
            agent.status = "BOUNDARY" if speed >= 0.15 else "PINNED"
            return
        if float(np.linalg.norm(plant.p - self.goal)) < plant.goal_tol:
            agent.done = True
            agent.status = "goal"
            return

        agent.stall_count = agent.stall_count + 1 if speed < 0.03 else 0
        if agent.stall_count > 60:
            agent.done = True
            agent.status = "STALLED"
            return

        u, u_proj, viol, info = self._compute_control_step(agent)
        if viol > 1e-6:
            agent.infeas_steps += 1

        # Telemetry at the decision instant. Radial and transverse split
        # against the nearest obstacle, QP correction du in normalized
        # inputs, projected input, feasibility residual, total field
        # energy, and the credited governor value as an observer metric
        # for every controller so the panels are comparable.
        v = plant.pdot
        vr = float(v @ n_hat)
        vp = float(np.linalg.norm(v - vr * n_hat))
        du = np.asarray(u_proj, float) - np.asarray(info["u_nom"], float)
        up = np.asarray(u_proj, float)
        E_tot = 0.0
        gov_min = np.inf
        for (c, R) in self.obstacles:
            s_i = float(np.linalg.norm(plant.p - c)) - (R + self.margin)
            _, E_i, _, _ = field_profiles(s_i, self.sigma, self.beta)
            E_tot += E_i
            gov_min = min(gov_min, governor_h(
                plant.p, plant.heading, speed, c, R + self.margin,
                self.a_brk(), self.gamma_perp, self.s_cap))
        agent.telemetry.append((
            agent.steps_taken * self.dt, plant.p[0], plant.p[1],
            v[0], v[1], clearance, vr, vp, speed, viol,
            float(info["n_rows"]), du[0], du[1], up[0], up[1],
            n_hat[0], n_hat[1], E_tot, gov_min))

        plant.step(self.dt, u, self._repair_circle_penetration)

        # Terminal contact is judged here, at the speed the shell was
        # actually touched with, before kill_inward zeroes it. Judging on
        # the next frame would misreport every fast contact as PINNED.
        c2, R2, d2, n2 = self.nearest_obstacle(plant.p)
        agent.min_clearance = min(agent.min_clearance, d2 - R2)
        if d2 <= R2:
            agent.done = True
            agent.status = "COLLISION"
            agent.contact_speed = plant.speed
        elif d2 <= R2 + self.margin + 1e-9:
            agent.done = True
            agent.contact_speed = plant.speed
            agent.status = "BOUNDARY" if plant.speed >= 0.15 else "PINNED"
            plant.kill_inward(n2)

        agent.last_u = np.asarray(u, float).copy()
        agent.path.append(plant.p.copy())
        agent.steps_taken += 1

    # ------------------- run control -------------------

    def _start_run(self, keys):
        if self.running:
            return
        self.running = True
        self._set_button_labels(running=True)
        self._clear_agent_artists()
        theta0 = self._initial_heading()
        self.active = []
        for k in keys:
            ag = self.agents[k]
            ag.reset(self.make_plant(), self.start, theta0,
                     self._initial_speed())
            self.active.append(ag)
        self.anim = FuncAnimation(self.fig, self._update_frame,
                                  frames=self.steps,
                                  interval=self.dt * 1000, blit=False,
                                  repeat=False)
        self.fig.canvas.draw_idle()

    def _stop_simulation(self):
        if self.anim is not None:
            self.anim.event_source.stop()
        self.running = False
        self.anim = None
        self._set_button_labels(running=False)
        for ag in self.active:
            print(f"{ag.label:6s} {ag.status:10s} "
                  f"steps={ag.steps_taken:4d}  "
                  f"min clearance={ag.min_clearance:6.3f} m  "
                  f"infeasible steps={ag.infeas_steps}")

    def _update_frame(self, frame_num):
        for ag in self.active:
            self._advance_agent(ag)
            path = np.array(ag.path)
            ag.line.set_data(path[:, 0], path[:, 1])
            ag.dot.set_data([ag.plant.p[0]], [ag.plant.p[1]])
            self._draw_agent_arrow(ag)
        self.info_text.set_text(self._format_status())
        if all(ag.done for ag in self.active) or frame_num >= self.steps - 1:
            self._stop_simulation()
        arts = []
        for ag in self.active:
            arts.extend([ag.line, ag.dot])
        arts.append(self.info_text)
        return tuple(arts)

    def _draw_agent_arrow(self, ag):
        vel = ag.plant.pdot
        if ag.quiv is None:
            ag.quiv = self.ax.quiver(
                [ag.plant.p[0]], [ag.plant.p[1]], [vel[0]], [vel[1]],
                angles="xy", scale_units="xy", scale=3.0, width=0.010,
                alpha=0.95, pivot="tail", color=ag.color)
        else:
            ag.quiv.set_offsets(np.array([[ag.plant.p[0], ag.plant.p[1]]]))
            ag.quiv.set_UVC([vel[0]], [vel[1]])

    def _format_status(self):
        lines = []
        for ag in self.active:
            p = ag.plant.p
            _cn, R_n, d_n, _nn = self.nearest_obstacle(p)
            gap = d_n - R_n
            lines.append(f"{ag.label:6s} v={ag.plant.speed:4.2f}  "
                         f"gap={gap:5.2f}  {ag.compute_ms:5.2f} ms  "
                         f"th={np.degrees(ag.plant.theta):6.1f}  "
                         f"{ag.status}")
        return "\n".join(lines)

    # --------- Diagnostics ---------

    def diagnose_current(self, save=None, show=False):
        """Open loop roll out of all three controllers on the current
        scenario and launch speed, with per step telemetry, a printed
        summary and the diagnostics figure."""
        grow = max(0.2, 0.15 * self.vmax)
        print(f"\n=== DIAGNOSE  scenario={self.scenario}"
              f"  vmax={self.vmax:.2f}  v0={self._initial_speed():.2f} ===")
        results = []
        for key, label, color, ls in CONTROLLER_SPECS:
            _, ag = self.run_open_loop_once(key)
            results.append((key, label, color, ls, ag))
            tel = telemetry_arrays(ag)
            reach = 2.0 * self.apf_sig if key == "apf" else None
            i0 = swerve_onset(tel, grow=grow, engage_gap=reach)
            onset = ("never" if i0 is None else
                     f"t={tel['t'][i0]:.2f}s at gap {tel['gap'][i0]:.2f}")
            print(f"  {label:6s} {ag.status:8s} "
                  f"t={ag.steps_taken * self.dt:5.2f}s"
                  f"  minGap={ag.min_clearance:6.3f}"
                  f"  infeas={ag.infeas_steps:3d}"
                  f"  swerve onset: {onset}")
        fig = plot_diagnostics(self, results, save=save)
        if show and fig is not None:
            try:
                fig.show()
            except Exception:
                pass
        return fig

    # --------- Non animated compare run ---------

    def run_open_loop_once(self, key, steps=None):
        """Headless roll out of one controller."""
        steps = steps or self.steps
        ag = Agent(key, key.upper(), "k", "-")
        ag.reset(self.make_plant(), self.start, self._initial_heading(),
                 self._initial_speed())
        for _ in range(steps):
            if ag.done:
                break
            self._advance_agent(ag)
        if not ag.done:
            ag.status = "timeout"
        return np.array(ag.path), ag

    def on_compare(self, _event=None):
        self._clear_agent_artists()
        rows = []
        for key, label, color, ls in CONTROLLER_SPECS:
            path, ag = self.run_open_loop_once(key)
            self.ax.plot(path[:, 0], path[:, 1], color=color, ls=ls,
                         lw=2.2, label=f"{label} (open loop)")
            k2, htv, L = _path_smoothness_metrics(path,
                                                  ds_floor=self.ds_floor)
            rows.append((label, k2, htv, L, ag))
        self.ax.legend(loc="upper left", fontsize=7, ncol=2)

        print(f"\n=== OPEN LOOP COMPARE  scenario={self.scenario}"
              f"  v0={self._initial_speed():.2f} ===")
        print(f"{'ctrl':6s} {'status':10s} {'t(s)':>6s} {'avg v':>6s} "
              f"{'L(m)':>7s} {'minGap':>7s} {'Sk2/L':>8s} {'infeas':>7s}")
        txt = [f"scenario {self.scenario}, v0 {self._initial_speed():.2f}"]
        for label, k2, htv, L, ag in rows:
            t = max(ag.steps_taken * self.dt, 1e-9)
            print(f"{label:6s} {ag.status:10s} {t:6.2f} {L / t:6.2f} "
                  f"{L:7.2f} {ag.min_clearance:7.3f} {k2:8.3f} "
                  f"{ag.infeas_steps:7d}")
            txt.append(f"{label:6s} {ag.status:9s} t={t:5.2f} "
                       f"avg v={L / t:4.2f} gap={ag.min_clearance:5.2f}")
        self.info_text.set_text("\n".join(txt))
        self.fig.canvas.draw_idle()


# ====================== Diagnostics ======================

_TEL_KEYS = ("t", "x", "y", "vx", "vy", "gap", "vr", "vp", "sp", "viol",
             "nbar", "dux", "duy", "upx", "upy", "nx", "ny", "E", "gov")


def telemetry_arrays(ag):
    """Agent telemetry as a dict of arrays, or None if empty. Fields:
    time, position, velocity, surface gap, radial velocity vr (positive
    receding), transverse speed vp, speed, feasibility residual viol,
    active row count, QP correction du in normalized inputs, projected
    input up, outward normal n, total field energy E, and the credited
    governor value gov recorded for every controller as an observer."""
    if not getattr(ag, "telemetry", None):
        return None
    T = np.asarray(ag.telemetry, float)
    return {k: T[:, i] for i, k in enumerate(_TEL_KEYS)}


def swerve_onset(tel, grow=0.3, engage_gap=None):
    """Index of the first sample, after the controller has engaged, where
    the transverse speed has grown by grow over its initial value while
    still moving. Engagement means a constraint row is active; for the
    barrier free APF, pass its field reach as engage_gap instead."""
    if tel is None or len(tel["t"]) < 3:
        return None
    engaged = tel["nbar"] > 0
    if engage_gap is not None and not engaged.any():
        engaged = tel["gap"] < engage_gap
    seen = np.cumsum(engaged.astype(int)) > 0
    cond = seen & (tel["vp"] > tel["vp"][0] + grow) & (tel["sp"] > 0.2)
    if not cond.any():
        return None
    return int(np.argmax(cond))


def plot_diagnostics(sim, results, save=None):
    """One figure answering four questions per controller: when does the
    transverse channel start being used (stars and the vp panel), does
    the field energy stay under the cap while speed is held (E and speed
    panels), was any certificate row infeasible (margin panel below
    zero), and what does the correction spend, brake or yaw (split)."""
    fig = plt.figure(figsize=(15.5, 8.5))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.35,
                          left=0.045, right=0.985, top=0.90, bottom=0.07)
    axT = fig.add_subplot(gs[:, :2])
    axVp = fig.add_subplot(gs[0, 2])
    axE = fig.add_subplot(gs[0, 3])
    axSp = fig.add_subplot(gs[1, 2])
    axMg = fig.add_subplot(gs[1, 3])
    axDu = fig.add_subplot(gs[2, 2])
    axGv = fig.add_subplot(gs[2, 3])

    axT.set_xlim(sim.xmin, sim.xmax)
    axT.set_ylim(sim.ymin, sim.ymax)
    axT.set_aspect("equal", "box")
    for (c, R) in sim.obstacles:
        axT.add_patch(plt.Circle(tuple(c), R, color="k", fill=False, lw=2))
        axT.add_patch(plt.Circle(tuple(c), R + sim.margin, color="k",
                                 fill=False, lw=0.8, ls=":", alpha=0.6))
    axT.plot([sim.start[0]], [sim.start[1]], "g^", ms=10)
    axT.plot([sim.goal[0]], [sim.goal[1]], "rx", ms=12, mew=2.2)

    grow = max(0.2, 0.15 * sim.vmax)
    sc_handle = None
    for (key, label, color, ls, ag) in results:
        tel = telemetry_arrays(ag)
        if tel is None:
            continue
        axT.plot(tel["x"], tel["y"], color=color, ls=ls, lw=1.6,
                 alpha=0.9, label=f"{label} ({ag.status})")
        ratio = tel["vp"] / np.maximum(tel["sp"], 1e-6)
        sc_handle = axT.scatter(tel["x"][::3], tel["y"][::3],
                                c=np.clip(ratio[::3], 0, 1),
                                cmap="viridis", vmin=0, vmax=1, s=10,
                                zorder=3)
        reach = 2.0 * sim.apf_sig if key == "apf" else None
        i0 = swerve_onset(tel, grow=grow, engage_gap=reach)
        if i0 is not None:
            axT.plot([tel["x"][i0]], [tel["y"][i0]], marker="*", ms=17,
                     color=color, mec="k", zorder=5)
            axVp.axvline(tel["t"][i0], color=color, lw=0.8, alpha=0.5)

        axVp.plot(tel["t"], tel["vp"], color=color, ls=ls)
        axE.plot(tel["t"], tel["E"], color=color, ls=ls)
        axSp.plot(tel["t"], tel["sp"], color=color, ls=ls)
        if (tel["nbar"] > 0).any():
            axMg.plot(tel["t"], -tel["viol"], color=color, ls=ls)
        axDu.plot(tel["t"], np.abs(tel["dux"]), color=color, ls="-",
                  lw=1.1)
        axDu.plot(tel["t"], np.abs(tel["duy"]), color=color, ls=":",
                  lw=1.4)
        axGv.plot(tel["t"], tel["gov"], color=color, ls=ls)

    axVp.set_title("transverse speed v_perp (m/s), line = onset",
                   fontsize=9)
    axE.set_title("field energy E, dashed = E_cap", fontsize=9)
    axE.axhline(sim._E_cap(), color="k", lw=0.8, ls="--")
    axSp.set_title("speed (m/s)", fontsize=9)
    axMg.set_title("row margin (below 0 = infeasible)", fontsize=9)
    axMg.axhline(0.0, color="k", lw=0.7)
    axDu.set_title("|correction| split: solid brake, dotted yaw",
                   fontsize=9)
    axGv.set_title("credited governor h_g (observer, all)", fontsize=9)
    axGv.axhline(0.0, color="k", lw=0.7)
    for ax in (axDu, axGv):
        ax.set_xlabel("t (s)", fontsize=8)
    if sc_handle is not None:
        fig.colorbar(sc_handle, ax=axT, fraction=0.04, pad=0.02,
                     label="v_perp / speed")
    axT.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"scenario {sim.scenario}   vmax {sim.vmax:.1f}   "
                 f"v0 {sim._initial_speed():.2f} m/s", fontsize=11)
    if save:
        fig.savefig(save, dpi=110)
        print(f"  diagnostics figure saved to {save}")
    return fig


# ====================== Headless drivers ======================

_ABBR = {"goal": "goal", "STALLED": "STAL", "PINNED": "PIN ",
         "BOUNDARY": "BNDY", "COLLISION": "COLL", "timeout": "TOUT"}


def run_matrix(v0_fracs=(0.3, 0.6, 1.0), brake=False, a_lat=None):
    """Status grid over every scenario and launch speed. A cell is a WIN
    when VCBF reaches the goal and neither baseline does."""
    wins = []
    print(f"\nbaseline brake row: {'on' if brake else 'off'}")
    for skey in SCENARIOS:
        sc = SCENARIOS[skey]
        print(f"\n=== scenario {skey}  vmax {sc['vmax']:.1f}  "
              f"({sc['blurb']}) ===")
        print(f"{'v0':>5} | {'CBF':>5} {'t':>6} | {'APF':>5} {'t':>6} | "
              f"{'VCBF':>5} {'t':>6} |")
        for f in v0_fracs:
            sim = InteractiveSimDrag(show=False)
            sim.apply_scenario(skey)
            sim.cbf_brake = brake
            if a_lat is not None:
                sim.a_lat = float(a_lat)
            sim.v0_frac = f
            st, tt = {}, {}
            for k, _l, _c, _s in CONTROLLER_SPECS:
                _, ag = sim.run_open_loop_once(k)
                st[k] = ag.status
                tt[k] = ag.steps_taken * sim.dt
            win = (st["vcbf"] == "goal" and st["cbf"] != "goal"
                   and st["apf"] != "goal")
            if win:
                wins.append((skey, f * sc["vmax"]))
            print(f"{f * sc['vmax']:5.2f} | "
                  f"{_ABBR.get(st['cbf'], st['cbf'])[:5]:>5} "
                  f"{tt['cbf']:6.2f} | "
                  f"{_ABBR.get(st['apf'], st['apf'])[:5]:>5} "
                  f"{tt['apf']:6.2f} | "
                  f"{_ABBR.get(st['vcbf'], st['vcbf'])[:5]:>5} "
                  f"{tt['vcbf']:6.2f} |"
                  + ("  <== VCBF only" if win else ""))
    print("\n=== VCBF only cells ===")
    if not wins:
        print("  none on this grid")
    for skey, v0 in wins:
        print(f"  scenario={skey:8s} v0={v0:.2f} m/s")
    return wins


def _speed_sweep(a_lat=None):
    """Hold the input bounds fixed and sweep the launch speed on the
    sprint geometry, showing the baseline's whole dial in one table:
    the plain HOCBF (late at speed), the same HOCBF with the brake row
    (never late, conservative at every speed), and ours. The stopping
    need grows as $v^2/(2 a_{max})$ and the friction limited turn radius
    as $v^2/a_{lat}$, while the linear class K margin of the standard
    HOCBF admits approach at range regardless."""
    print("\n=== sprint geometry, v0 = vmax sweep, bounds fixed ===")
    print(f"{'vmax':>5} | {'CBF':>5} {'minGap':>7} | "
          f"{'CBF+b':>5} {'minGap':>7} | {'VCBF':>5} {'minGap':>7} |")
    for vm in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
        row = f"{vm:5.1f} |"
        for key, brake in (("cbf", False), ("cbf", True), ("vcbf", False)):
            sim = InteractiveSimDrag(show=False)
            sim.apply_scenario("sprint")
            sim.vmax = vm
            sim.v0_frac = 1.0
            sim.cbf_brake = brake
            if a_lat is not None:
                sim.a_lat = float(a_lat)
            _, ag = sim.run_open_loop_once(key)
            row += (f" {_ABBR.get(ag.status, ag.status)[:5]:>5} "
                    f"{ag.min_clearance:7.3f} |")
        print(row)


# ====================== Self test ======================

def _car_deriv(state, a, w):
    x, y, th, s = state
    return np.array([s * np.cos(th), s * np.sin(th), w, a])


def _rk4(state, a, w, h):
    k1 = _car_deriv(state, a, w)
    k2 = _car_deriv(state + 0.5 * h * k1, a, w)
    k3 = _car_deriv(state + 0.5 * h * k2, a, w)
    k4 = _car_deriv(state + h * k3, a, w)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _selftest():
    """Headless checks: closed form profiles against finite differences,
    the analytic $\\ddot E$ and governor $\\dot h_g$ against numeric
    differentiation along exact car flows, then rollouts everywhere."""
    sigma, beta = 0.55, 1.0
    eps = 1e-6
    print("=== field profile derivative checks ===")
    ok = True
    for s in (0.05, 0.3, 0.7, 1.2, 2.0):
        _, E, dE, ddE = field_profiles(s, sigma, beta)
        _, Ep, dEp, _ = field_profiles(s + eps, sigma, beta)
        _, Em, dEm, _ = field_profiles(s - eps, sigma, beta)
        err = max(abs((Ep - Em) / (2 * eps) - dE),
                  abs((dEp - dEm) / (2 * eps) - ddE))
        ok = ok and err < 1e-5
        print(f"  s={s:4.2f}  max err vs finite diff = {err:.2e}")
    print("  profiles:", "PASS" if ok else "FAIL")

    print("\n=== analytic Eddot and governor hdot vs numeric flow ===")
    c = np.array([0.0, 0.0])
    R_eff = 0.9
    a_brk, gamma = 2.1, 2.5
    s_cap = 0.12
    rng = np.random.default_rng(3)
    ok2 = True
    for trial in range(6):
        th = rng.uniform(-np.pi, np.pi)
        state = np.array([rng.uniform(1.2, 2.2) * np.cos(th),
                          rng.uniform(1.2, 2.2) * np.sin(th),
                          rng.uniform(-np.pi, np.pi),
                          rng.uniform(0.8, 2.5)])
        a_in = rng.uniform(-2.0, 2.0)
        w_in = rng.uniform(-2.0, 2.0)

        def E_of(st):
            s_gap = float(np.linalg.norm(st[:2] - c)) - R_eff
            return field_profiles(s_gap, sigma, beta)[1]

        def Edot_of(st):
            p = st[:2]
            rel = p - c
            d = float(np.linalg.norm(rel))
            n = rel / d
            e = np.array([np.cos(st[2]), np.sin(st[2])])
            _, _, dE, _ = field_profiles(d - R_eff, sigma, beta)
            return dE * st[3] * float(n @ e)

        def gov_of(st):
            p = st[:2]
            e = np.array([np.cos(st[2]), np.sin(st[2])])
            return governor_h(p, e, st[3], c, R_eff, a_brk, gamma, s_cap)

        h = 1e-5
        st_p = _rk4(state, a_in, w_in, h)
        st_m = _rk4(state, a_in, w_in, -h)

        num_Eddot = (Edot_of(st_p) - Edot_of(st_m)) / (2 * h)
        p = state[:2]
        rel = p - c
        d = float(np.linalg.norm(rel))
        n = rel / d
        t = _perp(n)
        e = np.array([np.cos(state[2]), np.sin(state[2])])
        sp = state[3]
        v_r = sp * float(n @ e)
        v_p = sp * float(t @ e)
        _, _, dE, ddE = field_profiles(d - R_eff, sigma, beta)
        ana_Eddot = (ddE * v_r * v_r + (dE / d) * v_p * v_p
                     + dE * (float(n @ e) * a_in - v_p * w_in))
        errE = abs(num_Eddot - ana_Eddot)

        # Governor: compare numeric $\dot h_g$ with the analytic row
        # applied to the same raw inputs, away from the $v_r = 0$ kink.
        if v_r > -0.05:
            errG = 0.0
        else:
            num_hdot = (gov_of(st_p) - gov_of(st_m)) / (2 * h)
            av, bv, hg = governor_row(p, e, sp, c, R_eff, a_brk, gamma,
                                      1.0, s_cap, 1.0, 1.0)
            ana_hdot = float(av @ np.array([a_in, w_in])) - (bv + hg)
            # bv = -alpha*hg - const with alpha = 1, so const = -bv - hg
            errG = abs(num_hdot - ana_hdot)
        ok2 = ok2 and errE < 1e-4 and errG < 1e-4
        print(f"  trial {trial}: |Eddot err|={errE:.2e}  "
              f"|gov hdot err|={errG:.2e}")
    print("  rows:", "PASS" if ok2 else "FAIL")

    print("\n=== rollout statuses on every scenario ===")
    run_matrix(v0_fracs=(0.6, 1.0))


# ====================== Main ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="derivative and row checks plus rollouts")
    parser.add_argument("--matrix", action="store_true",
                        help="status grid over scenarios and speeds")
    parser.add_argument("--brake", action="store_true",
                        help="baseline CBF also gets the braking distance"
                             " row (toggleable in the GUI)")
    parser.add_argument("--alat", type=float, default=None,
                        help="override the lateral acceleration cap")
    parser.add_argument("--sweep", action="store_true",
                        help="vmax sweep on the sprint geometry")
    parser.add_argument("--diagnose", metavar="SCENARIO", nargs="?",
                        const="sprint", choices=list(SCENARIOS),
                        help="headless diagnostics for one scenario, PNG")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default=None,
                        help="scenario preset for the GUI start state")
    parser.add_argument("--v0", type=float, default=None,
                        help="initial speed in m/s, clipped to vmax")
    parser.add_argument("--vmax", type=float, default=None,
                        help="override the scenario top speed")
    parser.add_argument("--out", default=None,
                        help="output PNG path for --diagnose")
    args = parser.parse_args()

    headless = args.selftest or args.matrix or args.sweep or args.diagnose
    if headless:
        matplotlib.use("Agg")
        if args.selftest:
            _selftest()
        if args.matrix:
            run_matrix(brake=args.brake, a_lat=args.alat)
        if args.sweep:
            _speed_sweep(a_lat=args.alat)
        if args.diagnose:
            sim = InteractiveSimDrag(show=False)
            sim.apply_scenario(args.diagnose)
            sim.cbf_brake = args.brake
            if args.alat is not None:
                sim.a_lat = float(args.alat)
            if args.vmax is not None:
                sim.vmax = float(args.vmax)
            if args.v0 is not None:
                sim.v0_frac = float(np.clip(args.v0 / sim.vmax, 0.0, 1.0))
            out = args.out or f"diagnostics_{args.diagnose}.png"
            sim.diagnose_current(save=out, show=False)
    else:
        v0f = None
        if args.v0 is not None:
            vm = args.vmax if args.vmax is not None else \
                (SCENARIOS[args.scenario]["vmax"] if args.scenario else 2.5)
            v0f = args.v0 / max(vm, EPS)
        InteractiveSimDrag(scenario=args.scenario, v0_frac=v0f,
                           vmax=args.vmax, brake=args.brake,
                           a_lat_override=args.alat)