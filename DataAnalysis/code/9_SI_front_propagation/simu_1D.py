import numpy as np
import tqdm
import copy
import matplotlib.pyplot as plt


from dendrogenesis import project_style


project_style.set_style()
_ARFIMA_KERNEL_CACHE = {}


def _get_arfima_fft_kernel(d: float, total_length: int):
    key = (round(float(d), 12), int(total_length))
    if key in _ARFIMA_KERNEL_CACHE:
        return _ARFIMA_KERNEL_CACHE[key]

    if abs(d) < 1e-8:
        w = np.zeros(total_length, dtype=float)
        w[0] = 1.0
    else:
        w = np.empty(total_length, dtype=float)
        w[0] = 1.0
        for j in range(1, total_length):
            w[j] = w[j - 1] * (j - 1 + d) / j

    fft_length = 1 if total_length <= 1 else 1 << ((2 * total_length - 2).bit_length())
    kernel_fft = np.fft.rfft(w, n=fft_length)
    _ARFIMA_KERNEL_CACHE[key] = (kernel_fft, fft_length)
    return kernel_fft, fft_length


def arfima_bank(
        d: float,
        arfima_sigma_eps, 
        horizon: int,
        dt: float,
        n_trajs: int,
):
    """
    Generate n_trajs independent ARFIMA(0,d,0) series of length horizon using the time-varying sigma schedule arfima_sigma_eps.
    Returns array of shape (n_trajs, horizon).
    """
    arfima_sigma_eps = np.asarray(arfima_sigma_eps, dtype=float)
    burnin = max(1024, horizon // 2)
    N = horizon + burnin 
    #sigma is interpreted as the 1-minute increment scale. At time step dt,
    #ARFIMA increments scale like dt^(alpha/2) = dt^(1/2 + d).
    increment_scale = float(dt) ** (0.5 + float(d))

    #Innovations eps, vectorized over trajectories
    eps = np.empty((n_trajs, N), dtype=float)

    #burn-in with constant sigma
    eps[:, :burnin] = np.random.normal(
        loc=0.0,
        scale=arfima_sigma_eps[0] * increment_scale,
        size=(n_trajs, burnin),
    )

    #main horizon part with time-dependent sigma
    eps[:, burnin:] = np.random.normal(
        loc=0.0,
        scale=(arfima_sigma_eps * increment_scale)[None, :],
        size=(n_trajs, horizon),
    )

    #FFT-based convolution
    W, L = _get_arfima_fft_kernel(d, N)
    E = np.fft.rfft(eps, n=L, axis=-1) #(n_trajs, L/2+1)
    y_full = np.fft.irfft(E * W, n=L, axis=-1)[..., :N]  

    y = y_full[..., burnin:] #(n_trajs, horizon)
    return y

class ArfimaTrajectoryPool:
    """
    Pool of ARFIMA trajectories.

    - Generates trajectories in chunks of size chunk_size:
        bank.shape = (chunk_size, horizon)
    - get(n) returns n independent ARFIMA paths (n, horizon).
    - When the current bank is exhausted, a new one is generated.
    """

    def __init__(self, d, sigma_eps, horizon, dt, chunk_size=256):
        self.d = float(d)
        self.sigma_eps = np.asarray(sigma_eps, dtype=float)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.chunk_size = int(chunk_size)

        self._bank = None
        self._next_row = 0

    def _refill(self):
        self._bank = arfima_bank(
            d=self.d,
            arfima_sigma_eps=self.sigma_eps,
            horizon=self.horizon,
            dt=self.dt,
            n_trajs=self.chunk_size,
        )
        self._next_row = 0

    def get(self, n: int) -> np.ndarray:
        """
        Return n new independent ARFIMA trajectories, shape (n, horizon).
        """
        n = int(n)
        if n <= 0:
            return np.empty((0, self.horizon), dtype=float)

        #If n is large, generate directly in one shot
        if n >= self.chunk_size:
            return arfima_bank(
                d=self.d,
                arfima_sigma_eps=self.sigma_eps,
                horizon=self.horizon,
                dt=self.dt,
                n_trajs=n,
            )

        if self._bank is None or self._next_row + n > self._bank.shape[0]:
            self._refill()

        out = self._bank[self._next_row:self._next_row + n, :]
        self._next_row += n
        return out




def simu_tip_splitting(
        r,
        sigma,
        alpha,
        T_max=300,
        dt=1,
        top_n=None,
        with_angle=None,
        verbose=False,
        with_starting_point=False,
        length_start=20,
        take_abs_positions=True,
):
    """
    """
    all_ts = np.arange(0, T_max, step=dt)

    r_interp = np.interp(all_ts, r[0], r[1])
    sigma_interp = np.interp(all_ts, sigma[0], sigma[1])
    alpha_interp = np.interp(all_ts, alpha[0], alpha[1])

    list_positions = []
    list_tip_id = []

    positions = np.array([length_start], dtype=np.float64)
    angles = np.array([0], dtype=np.float64)
    length = copy.deepcopy(positions)
    tip_id = np.array([0])
    next_tip_id = 1

    for i, t in tqdm.tqdm(enumerate(all_ts), disable=not(verbose)):
        rr = r_interp[i]
        s = sigma_interp[i]
        a = alpha_interp[i]

        increments = np.random.normal(0, s * np.sqrt(dt), size=positions.shape[0])
        length += increments
        positions += increments * np.cos(angles)

        if with_starting_point:
            idx_keep = length > 0
            positions = positions[idx_keep]
            angles = angles[idx_keep]
            length = length[idx_keep]
            tip_id = tip_id[idx_keep]
            if positions.shape[0] == 0:
                positions = np.array([length_start], dtype=np.float64)
                angles = np.array([0], dtype=np.float64)
                length = copy.deepcopy(positions)
                tip_id = np.array([0])

        if take_abs_positions:
            positions = np.abs(positions)

        poisson_intensity = dt * rr * np.ones_like(positions)
        n_new_branch = np.random.poisson(poisson_intensity)
        max_new = int(np.max(n_new_branch)) if n_new_branch.shape[0] > 0 else 0

        if max_new > 0:
            new_positions = np.repeat(np.expand_dims(positions, axis=1), max_new, axis=1)
            new_angles = np.repeat(np.expand_dims(angles, axis=1), max_new, axis=1)

            idx = np.repeat(np.expand_dims(np.arange(max_new), axis=0), positions.shape[0], axis=0)
            idx_keep = idx < np.expand_dims(n_new_branch, axis=1)

            new_positions = new_positions[idx_keep]
            new_angles = new_angles[idx_keep]
            new_length = np.zeros_like(new_positions)

            angles = np.concatenate([angles, new_angles])
            positions = np.concatenate([positions, new_positions])
            length = np.concatenate([length, new_length])
            tip_id = np.concatenate([tip_id, next_tip_id + np.arange(new_positions.shape[0])])
            next_tip_id = np.max(tip_id) + 1

        if not(top_n is None):
            idx_keep = np.argsort(positions)[-top_n:]
            positions = positions[idx_keep]
            angles = angles[idx_keep]
            length = length[idx_keep]
            tip_id = tip_id[idx_keep]

        list_positions.append(copy.deepcopy(positions))
        list_tip_id.append(copy.deepcopy(tip_id))

    return(all_ts, list_positions, list_tip_id)

def simu_arfima(
        lamb, sigma, alpha,
        T_max=300,
        dt=1,
        top_n=None,
        with_angle=None,
        verbose=False,
        with_starting_point=False,
        arfima_chunk_size = 2048,
        with_death = False,
        birth_length = None,
        initial_length = 2,
        take_abs_positions = True,
        branching_uses_abs_positions = False,
):
    """
    ARFIMA-based version of simu.
    """

    all_ts = np.arange(0, T_max, step=dt)
    horizon = all_ts.shape[0]

    lamb_interp  = np.interp(all_ts, lamb[0],  lamb[1])
    sigma_interp = np.interp(all_ts, sigma[0], sigma[1])
    alpha_interp = np.interp(all_ts, alpha[0], alpha[1])

    #alpha = 1 + 2d
    alpha0 = float(alpha_interp[0])   #assume alpha is constant
    d = 0.5 * (alpha0 - 1.0)
    use_random_walk = np.isclose(alpha0, 1.0)

    #sigma is the 1-minute increment scale of the branch displacement.
    #arfima_bank rescales it to the current dt with dt^(alpha/2).
    sigma_eps = sigma_interp

    if not use_random_walk:
        #ARFIMA trajectory pool
        pool = ArfimaTrajectoryPool(
            d=d,
            sigma_eps=sigma_eps,
            horizon=horizon,
            dt=dt,
            chunk_size=arfima_chunk_size,
        )

    list_positions = []
    list_tip_id = []

    #Initial tip
    positions = np.array([initial_length], dtype=np.float64)
    angles    = np.array([0.0], dtype=np.float64)
    length    = positions.copy()
    tip_id    = np.array([0], dtype=np.int64)
    next_tip_id = 1

    if not use_random_walk:
        inc_per_branch = pool.get(1)  #shape (1, horizon)

    for i, t in tqdm.tqdm(enumerate(all_ts), disable=not(verbose)):
        l = lamb_interp[i]

        if use_random_walk:
            increments = np.random.normal(
                loc=0.0,
                scale=sigma_interp[i] * np.sqrt(dt),
                size=positions.shape[0],
            )
        else:
            #ARFIMA increment for each branch at this time step
            increments = inc_per_branch[:, i]  #shape (n_branches,)

        #Update length and radial position (same as in simu, but using ARFIMA increments)
        length   += increments
        positions += increments * np.cos(angles)

        if with_starting_point:
            #Kill tips whose length <= 0
            idx_keep = length > 0
            positions = positions[idx_keep]
            angles    = angles[idx_keep]
            length    = length[idx_keep]
            tip_id    = tip_id[idx_keep]
            if not use_random_walk:
                inc_per_branch = inc_per_branch[idx_keep, :]
            
            #If all tips died, restart a single one at length=2
            if positions.shape[0] == 0:
                if not(with_death):
                    positions = np.array([2.0], dtype=np.float64)
                    angles    = np.array([0.0], dtype=np.float64)
                    length    = positions.copy()
                    tip_id    = np.array([0], dtype=np.int64)
                    next_tip_id = 1
                    if not use_random_walk:
                        inc_per_branch = pool.get(1)
                else:
                    return("death",t)
        
        if take_abs_positions:
            positions = np.abs(positions)

        #Branching intensity
        if with_starting_point:
            poisson_intensity = dt * np.abs(length) * l
        else:
            branching_positions = np.abs(positions) if branching_uses_abs_positions else positions
            poisson_intensity = dt * branching_positions * l

        if not(with_angle is None) and (not with_starting_point):
            poisson_intensity /= 2.0

        n_new_branch = np.random.poisson(poisson_intensity)
        max_new = int(np.max(n_new_branch))

        if max_new > 0:
            #New positions along the existing segment
            if with_starting_point:
                #new positions along the segment from soma to tip
                new_positions = (
                    np.expand_dims(positions, axis=1)
                    - np.expand_dims(np.cos(angles), axis=1)
                    * np.random.random(size=(positions.shape[0], max_new))
                    * np.expand_dims(length, axis=1)
                )
                new_positions = np.abs(new_positions)
            else:
                #uniform along radial length
                new_positions = (
                    np.random.random(size=(positions.shape[0], max_new))
                    * np.expand_dims(positions, axis=1)
                )

            #Angles of new branches
            if with_angle is None:
                new_angles = np.zeros(new_positions.shape)
            elif with_angle == "uniform":
                new_angles = np.random.random(size=new_positions.shape) * 2 * np.pi
            elif with_angle == "orthogonal":
                new_angles = np.ones(new_positions.shape) * (np.pi / 2.0)*(2*np.random.randint(2) - 1)
            elif with_angle == "normal":
                new_angles = np.random.normal(
                    np.pi / 2.0, np.pi / 4.0, size=new_positions.shape
                )*(2*np.random.randint(2) - 1)
            else:
                raise ValueError(f"Unknown with_angle mode: {with_angle}")

            new_angles = np.expand_dims(angles, axis=1) + new_angles

            #Keep only the first n_new_branch[j] children of parent j
            idx = np.repeat(
                np.expand_dims(np.arange(max_new), axis=0),
                positions.shape[0],
                axis=0,
            )
            idx_keep = idx < np.expand_dims(n_new_branch, axis=1)

            #if with_starting_point and with_angle=="orthogonal":
            #    #branches that grow towards the interior should not contribute ?
            #    idx_keep = idx_keep*(np.cos(new_angles) >= 0)


            new_positions = new_positions[idx_keep]
            new_angles    = new_angles[idx_keep]

            #New branches start with length 0
            new_length = np.zeros_like(new_positions)

            #Number of new branches
            n_new_total = new_positions.shape[0]

            if n_new_total > 0:
                n_existing = positions.shape[0]
                new_tip_id = next_tip_id + np.arange(n_new_total, dtype=np.int64)

                if top_n is not None and n_existing + n_new_total > top_n:
                    combined_positions = np.concatenate([positions, new_positions])
                    idx_keep = np.argpartition(combined_positions, -top_n)[-top_n:]

                    keep_existing = idx_keep[idx_keep < n_existing]
                    keep_new = idx_keep[idx_keep >= n_existing] - n_existing

                    positions = positions[keep_existing]
                    angles = angles[keep_existing]
                    length = length[keep_existing]
                    tip_id = tip_id[keep_existing]
                    if not use_random_walk:
                        inc_per_branch = inc_per_branch[keep_existing, :]

                    new_positions = new_positions[keep_new]
                    new_angles = new_angles[keep_new]
                    new_length = new_length[keep_new]
                    new_tip_id = new_tip_id[keep_new]
                    n_new_total = int(new_positions.shape[0])

                if n_new_total > 0:
                    #Append only the branches that survive the top_n cutoff.
                    angles = np.concatenate([angles, new_angles])
                    positions = np.concatenate([positions, new_positions])
                    length = np.concatenate([length, new_length])
                    if not use_random_walk:
                        new_inc = pool.get(n_new_total)  #(n_new_total, horizon)
                        inc_per_branch = np.concatenate([inc_per_branch, new_inc], axis=0)

                    tip_id = np.concatenate([tip_id, new_tip_id])
                    next_tip_id = int(np.max(tip_id) + 1)

        #Top n pruning by position
        if top_n is not None and positions.shape[0] > top_n:
            idx_keep = np.argsort(positions)[-top_n:]
            positions = positions[idx_keep]
            angles    = angles[idx_keep]
            length    = length[idx_keep]
            tip_id    = tip_id[idx_keep]
            if not use_random_walk:
                inc_per_branch = inc_per_branch[idx_keep, :]

        list_positions.append(positions.copy())
        list_tip_id.append(tip_id.copy())

    return all_ts, list_positions, list_tip_id


def simu_arfima_front_only(
        lamb, sigma, alpha,
        T_max=300,
        dt=1,
        top_n=None,
        verbose=False,
        with_starting_point=True,
        length_start=2.0,
):
    """
    optimized arfima simulation for scaling sweeps.
    """
    all_ts = np.arange(0, T_max, step=dt)
    horizon = all_ts.shape[0]

    lamb_interp = np.interp(all_ts, lamb[0], lamb[1])
    sigma_interp = np.interp(all_ts, sigma[0], sigma[1])
    alpha_interp = np.interp(all_ts, alpha[0], alpha[1])

    alpha0 = float(alpha_interp[0])
    d = 0.5 * (alpha0 - 1.0)
    use_random_walk = np.isclose(alpha0, 1.0)
    sigma_eps = sigma_interp

    def _make_increment_suffixes(start_idx, n_trajs):
        remaining = horizon - int(start_idx)
        if n_trajs <= 0:
            return []
        if remaining <= 0:
            return [np.empty(0, dtype=float) for _ in range(n_trajs)]
        bank = arfima_bank(
            d=d,
            arfima_sigma_eps=sigma_eps[start_idx:],
            horizon=remaining,
            dt=dt,
            n_trajs=n_trajs,
        )
        return [bank[k] for k in range(n_trajs)]

    positions = np.array([length_start], dtype=np.float64)
    length = positions.copy()
    fronts = np.empty(horizon, dtype=np.float64)

    if not use_random_walk:
        branch_suffixes = _make_increment_suffixes(0, 1)
        branch_offsets = np.zeros(1, dtype=np.int64)

    for i, _t in tqdm.tqdm(enumerate(all_ts), total=horizon, disable=not(verbose)):
        l = lamb_interp[i]

        if use_random_walk:
            increments = np.random.normal(
                loc=0.0,
                scale=sigma_interp[i] * np.sqrt(dt),
                size=positions.shape[0],
            )
        else:
            increments = np.fromiter(
                (seq[offset] for seq, offset in zip(branch_suffixes, branch_offsets)),
                dtype=float,
                count=len(branch_suffixes),
            )
            branch_offsets = branch_offsets + 1

        length += increments
        positions += increments

        if with_starting_point:
            idx_keep = length > 0
            positions = positions[idx_keep]
            length = length[idx_keep]
            if not use_random_walk:
                keep_idx = np.flatnonzero(idx_keep)
                branch_suffixes = [branch_suffixes[k] for k in keep_idx]
                branch_offsets = branch_offsets[keep_idx]

            if positions.shape[0] == 0:
                positions = np.array([length_start], dtype=np.float64)
                length = positions.copy()
                if not use_random_walk:
                    branch_suffixes = _make_increment_suffixes(i + 1, 1)
                    branch_offsets = np.zeros(1, dtype=np.int64)

        positions = np.abs(positions)

        if with_starting_point:
            poisson_intensity = dt * np.abs(length) * l
        else:
            poisson_intensity = dt * positions * l

        n_new_branch = np.random.poisson(poisson_intensity)
        max_new = int(np.max(n_new_branch)) if n_new_branch.size else 0

        if max_new > 0:
            parent_idx = np.repeat(np.arange(positions.shape[0]), n_new_branch)
            n_new_total = int(parent_idx.size)

            if n_new_total > 0:
                n_existing = positions.shape[0]
                rand_u = np.random.random(size=n_new_total)
                if with_starting_point:
                    new_positions = np.abs(
                        positions[parent_idx] - rand_u * length[parent_idx]
                    )
                else:
                    new_positions = rand_u * positions[parent_idx]

                if top_n is not None and n_existing + n_new_total > top_n:
                    combined_positions = np.concatenate([positions, new_positions])
                    idx_keep = np.argpartition(combined_positions, -top_n)[-top_n:]

                    keep_existing = idx_keep[idx_keep < n_existing]
                    keep_new = idx_keep[idx_keep >= n_existing] - n_existing

                    positions = positions[keep_existing]
                    length = length[keep_existing]
                    if not use_random_walk:
                        branch_suffixes = [branch_suffixes[k] for k in keep_existing]
                        branch_offsets = branch_offsets[keep_existing]

                    new_positions = new_positions[keep_new]
                    n_new_total = int(new_positions.shape[0])

                if n_new_total > 0:
                    positions = np.concatenate([positions, new_positions])
                    length = np.concatenate([length, np.zeros_like(new_positions)])
                    if not use_random_walk:
                        branch_suffixes.extend(_make_increment_suffixes(i + 1, n_new_total))
                        branch_offsets = np.concatenate(
                            [branch_offsets, np.zeros(n_new_total, dtype=np.int64)]
                        )

        if top_n is not None and positions.shape[0] > top_n:
            idx_keep = np.argpartition(positions, -top_n)[-top_n:]
            positions = positions[idx_keep]
            length = length[idx_keep]
            if not use_random_walk:
                branch_suffixes = [branch_suffixes[k] for k in idx_keep]
                branch_offsets = branch_offsets[idx_keep]

        fronts[i] = float(np.max(positions))

    return all_ts, fronts

