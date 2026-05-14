"""
Exact 1D Riemann solver — outputs a CSV for comparison with the FVM solver.
Based on Toro (2009), ported from Philip Mocz's interactive version.

Usage:
    python analytic.py --problem sod --t 0.2 --N 400 --out analytic_sod.csv
    python analytic.py --problem lax --t 0.13 --N 400 --out analytic_lax.csv
"""

import argparse
import numpy as np


PROBLEMS = {
    "sod":    dict(rho_L=1.0,   vx_L=0.0,   P_L=1.0,
                   rho_R=0.125, vx_R=0.0,   P_R=0.1,   gamma=5/3),
    "lax":    dict(rho_L=0.445, vx_L=0.698, P_L=3.528,
                   rho_R=0.5,   vx_R=0.0,   P_R=0.571, gamma=5/3),
    "vacuum": dict(rho_L=1.0,   vx_L=-2.0,  P_L=0.4,
                   rho_R=1.0,   vx_R=2.0,   P_R=0.4,   gamma=5/3),
}


class ExactRiemann:
    """Exact Riemann solver for the 1D Euler equations (Toro 2009)."""

    def __init__(self, rho_L, vx_L, P_L, rho_R, vx_R, P_R, gamma):
        self.rho_L = rho_L; self.vx_L = vx_L; self.P_L = P_L
        self.rho_R = rho_R; self.vx_R = vx_R; self.P_R = P_R
        self.gam   = gamma

        g = gamma
        self.g1 = (g - 1) / (2 * g)
        self.g2 = (g + 1) / (2 * g)
        self.g3 = 2 * g / (g - 1)
        self.g4 = 2 / (g - 1)
        self.g5 = 2 / (g + 1)
        self.g6 = (g - 1) / (g + 1)
        self.g7 = (g - 1) / 2
        self.g8 = g - 1
        self.g9 = 1 / g

        self.cL = np.sqrt(g * P_L / rho_L)
        self.cR = np.sqrt(g * P_R / rho_R)

    def _f_and_df(self, P, rho_k, P_k, c_k):
        if P <= P_k:
            q  = P / P_k
            f  = self.g4 * c_k * (q**self.g1 - 1.0)
            df = (1.0 / (rho_k * c_k)) * q**(-self.g2)
        else:
            ak  = self.g5 / rho_k
            bk  = self.g6 * P_k
            qrt = np.sqrt(ak / (bk + P))
            f   = (P - P_k) * qrt
            df  = (1.0 - 0.5 * (P - P_k) / (bk + P)) * qrt
        return f, df

    def _guess_P(self):
        cup  = 0.25 * (self.rho_L + self.rho_R) * (self.cL + self.cR)
        P_pv = 0.5 * (self.P_L + self.P_R) + 0.5 * (self.vx_L - self.vx_R) * cup
        P_pv = max(0.0, P_pv)
        P_min, P_max = min(self.P_L, self.P_R), max(self.P_L, self.P_R)
        q = P_max / P_min
        if q < 2.0 and P_min < P_pv < P_max:
            return P_pv
        if P_pv < P_min:
            Pq  = (self.P_L / self.P_R)**self.g1
            vxm = (Pq * self.vx_L / self.cL + self.vx_R / self.cR
                   + self.g4 * (Pq - 1.0)) / (Pq / self.cL + 1.0 / self.cR)
            ptL = 1.0 + self.g7 * (self.vx_L - vxm) / self.cL
            ptR = 1.0 + self.g7 * (vxm - self.vx_R) / self.cR
            return 0.5 * (self.P_L * ptL**self.g3 + self.P_R * ptR**self.g3)
        geL = np.sqrt((self.g5 / self.rho_L) / (self.g6 * self.P_L + P_pv))
        geR = np.sqrt((self.g5 / self.rho_R) / (self.g6 * self.P_R + P_pv))
        return (geL * self.P_L + geR * self.P_R - (self.vx_R - self.vx_L)) / (geL + geR)

    def _star(self):
        P  = self._guess_P()
        vd = self.vx_R - self.vx_L
        for _ in range(100):
            fL, dfL = self._f_and_df(P, self.rho_L, self.P_L, self.cL)
            fR, dfR = self._f_and_df(P, self.rho_R, self.P_R, self.cR)
            dP = (fL + fR + vd) / (dfL + dfR)
            P  = max(1e-10, P - dP)
            if 2 * abs(dP) / (P + P + dP) < 1e-8:
                break
        vx = 0.5 * (self.vx_L + self.vx_R + fR - fL)
        return P, vx

    def sample(self, s, P_star, vx_star):
        """Evaluate (rho, vx, P) at similarity variable s = (x - x_disc) / t."""
        if s <= vx_star:
            if P_star <= self.P_L:                       # left rarefaction
                sh_L = self.vx_L - self.cL
                if s <= sh_L:
                    return self.rho_L, self.vx_L, self.P_L
                cmL  = self.cL * (P_star / self.P_L)**self.g1
                st_L = vx_star - cmL
                if s > st_L:
                    rho = self.rho_L * (P_star / self.P_L)**self.g9
                    return rho, vx_star, P_star
                vx  = self.g5 * (self.cL + self.g7 * self.vx_L + s)
                c   = self.g5 * (self.cL + self.g7 * (self.vx_L - s))
                rho = self.rho_L * (c / self.cL)**self.g4
                P   = self.P_L   * (c / self.cL)**self.g3
                return rho, vx, P
            else:                                         # left shock
                P_starL = P_star / self.P_L
                s_L = self.vx_L - self.cL * np.sqrt(self.g2 * P_starL + self.g1)
                if s <= s_L:
                    return self.rho_L, self.vx_L, self.P_L
                rho = self.rho_L * (P_starL + self.g6) / (P_starL * self.g6 + 1.0)
                return rho, vx_star, P_star
        else:
            if P_star > self.P_R:                        # right shock
                P_starR = P_star / self.P_R
                s_R = self.vx_R + self.cR * np.sqrt(self.g2 * P_starR + self.g1)
                if s >= s_R:
                    return self.rho_R, self.vx_R, self.P_R
                rho = self.rho_R * (P_starR + self.g6) / (P_starR * self.g6 + 1.0)
                return rho, vx_star, P_star
            else:                                         # right rarefaction
                sh_R = self.vx_R + self.cR
                if s >= sh_R:
                    return self.rho_R, self.vx_R, self.P_R
                cmR  = self.cR * (P_star / self.P_R)**self.g1
                st_R = vx_star + cmR
                if s <= st_R:
                    rho = self.rho_R * (P_star / self.P_R)**self.g9
                    return rho, vx_star, P_star
                vx  = self.g5 * (-self.cR + self.g7 * self.vx_R + s)
                c   = self.g5 * (self.cR  - self.g7 * (self.vx_R - s))
                rho = self.rho_R * (c / self.cR)**self.g4
                P   = self.P_R   * (c / self.cR)**self.g3
                return rho, vx, P

    def solve(self, x, t, x_disc=0.5):
        """Return arrays (rho, vx, P) on grid x at time t."""
        P_star, vx_star = self._star()
        rho = np.zeros(len(x))
        vx  = np.zeros(len(x))
        P   = np.zeros(len(x))
        for i, xi in enumerate(x):
            s = (xi - x_disc) / t
            rho[i], vx[i], P[i] = self.sample(s, P_star, vx_star)
        return rho, vx, P


def main():
    ap = argparse.ArgumentParser(description="Exact Riemann solver → CSV")
    ap.add_argument("--problem", default="sod", choices=PROBLEMS.keys())
    ap.add_argument("--t",   type=float, default=0.2)
    ap.add_argument("--N",   type=int,   default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg  = PROBLEMS[args.problem]
    rs   = ExactRiemann(**cfg)
    x    = np.linspace(0.5 / args.N, 1.0 - 0.5 / args.N, args.N)   # cell centres
    rho, vx, P = rs.solve(x, args.t)
    gam = cfg["gamma"]
    E   = P / (gam - 1) + 0.5 * rho * vx**2

    fname = args.out or f"analytic_{args.problem}.csv"
    header = "x,rho,vx,P,E"
    data   = np.column_stack([x, rho, vx, P, E])
    np.savetxt(fname, data, delimiter=",", header=header, comments="")
    print(f"Wrote {fname}  (N={args.N}, t={args.t}, problem={args.problem})")


if __name__ == "__main__":
    main()
