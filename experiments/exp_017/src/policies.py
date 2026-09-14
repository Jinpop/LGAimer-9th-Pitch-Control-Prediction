"""exp_017 — iteration-count carry policies, defined once and reused.

Every policy takes ONLY curves of folds that are strictly earlier than the fold
being predicted.  No policy is allowed to look at the curve it is scored on;
the oracle (argmin of the target fold) is computed separately and used as an
upper-reference only.

A "curve" is {iter:int -> valid_all Brier:float} on the shared exp_016 grid.
"""


def nearest_grid_iter(grid, k):
    """Identical tie-breaking to exp_016 train.py (abs distance, then value)."""
    return min(grid, key=lambda g: (abs(g - k), g))


def argmin_of(curve):
    k = min(curve, key=lambda x: curve[x])
    return int(k), float(curve[k])


# --------------------------------------------------------------------------- #
def p4_argmin(prev):
    """Current policy: previous fold's valid_all Brier argmin, no scaling."""
    return argmin_of(prev)[0]


def p5_flat(prev, eps, contiguous=True):
    """Median iter of the near-optimal band {C <= min + eps}.

    contiguous=True  -> the maximal run of grid points around the argmin that
                        stays inside the band (faithful to config's "구간").
    contiguous=False -> median over every grid point inside the band.
    """
    grid = sorted(prev)
    m = min(prev.values())
    inside = [k for k in grid if prev[k] <= m + eps]
    if not inside:
        return argmin_of(prev)[0]
    if not contiguous:
        return inside[len(inside) // 2]
    a = argmin_of(prev)[0]
    i = grid.index(a)
    lo = i
    while lo - 1 >= 0 and prev[grid[lo - 1]] <= m + eps:
        lo -= 1
    hi = i
    while hi + 1 < len(grid) and prev[grid[hi + 1]] <= m + eps:
        hi += 1
    run = grid[lo:hi + 1]
    return run[len(run) // 2]


def p5_upper(prev, eps):
    """Upper edge of the near-optimal band around the argmin.

    Motivated by an observation, not by the config: on these curves the argmin
    moves UP from fold to fold in almost every arm, so the two sides of the
    band are not symmetric in risk.  Taking the top of the band instead of its
    median is the cheapest way to lean the choice in the observed direction.
    """
    grid = sorted(prev)
    m = min(prev.values())
    a = argmin_of(prev)[0]
    i = grid.index(a)
    hi = i
    while hi + 1 < len(grid) and prev[grid[hi + 1]] <= m + eps:
        hi += 1
    return grid[hi]


def p10_mult(prev, c):
    """Previous argmin times a constant.  c must be chosen from folds that are
    strictly earlier than the one being scored."""
    return max(1, int(round(argmin_of(prev)[0] * float(c))))


def p6_plateau(prev, K, delta):
    """First iter after which the next K rounds buy less than delta."""
    grid = sorted(prev)
    for k in grid:
        ahead = [prev[j] for j in grid if k < j <= k + K]
        if not ahead:
            continue
        if prev[k] - min(ahead) < delta:
            return k
    return grid[-1]


def p7_scale(prev, n_prev, n_cur):
    """Previous argmin scaled by the training-row ratio (exp_006 P2 revisited)."""
    return int(round(argmin_of(prev)[0] * (float(n_cur) / float(n_prev))))


def p8_trend_lin(prevprev, prev):
    """Linear extrapolation through the two most recent argmins."""
    a0 = argmin_of(prevprev)[0]
    a1 = argmin_of(prev)[0]
    return max(1, int(round(2 * a1 - a0)))


def p8_trend_geo(prevprev, prev):
    """Geometric extrapolation through the two most recent argmins."""
    a0 = argmin_of(prevprev)[0]
    a1 = argmin_of(prev)[0]
    if a0 <= 0:
        return a1
    return max(1, int(round(a1 * (float(a1) / float(a0)))))
