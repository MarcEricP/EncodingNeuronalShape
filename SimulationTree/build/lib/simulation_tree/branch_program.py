import math
import numpy as np
from numba.typed import List, Dict
from scipy.stats import laplace_asymmetric,gennorm,expon
from scipy import special
from numba import types
import numpy as np
from numba import njit

PROGRAM_PALAVALLI = 0  # linear phases without dependency between speed and duration
PROGRAM_ARFIMA    = 1  # fractional differencing weights psi (len M+1)
PROGRAM_GENNORM = 2 # IID steps

def to_numba_float_list(py_list):
    nb_list = List.empty_list(types.float64)
    for x in py_list:
        nb_list.append(float(x))
    return(nb_list)

def get_next_increment(
    program_type: int,
    tip_keys,          # numba.List of tip_id (lineage)
    prog_data,         # Dict[int -> float64[:]]
    is_contact_flags,  
    # common
    horizon: int,
    dt: float,
    # contact:
    # contact_penality:int,
    contact_memory_time:int,#time for which the contact memory is kept
    contact_mean_retraction : float,#mean retraction after contact
    contact_std_retraction: float,#std retraction after contact
    # GENORM PARAMS
    gennorm_beta,#array with the value taken by the parameter for the next horizon steps
    gennorm_loc ,#array with the value taken by the parameter for the next horizon steps
    gennorm_scale ,#array with the value taken by the parameter for the next horizon steps
    # ARFIMA params/state
    arfima_d,
    arfima_kappa_eps,#array with the value taken by the parameter for the next horizon steps
    arfima_scale_eps,#array with the value taken by the parameter for the next horizon steps
    arfima_loc_eps:float,
    arfima_p_mb:float,
    arfima_d_mb:float,
    # Palavalli telegraph params/state
    pal_von,#array with the value taken by the parameter for the next horizon steps
    pal_voff,#array with the value taken by the parameter for the next horizon steps
    pal_kon,#array with the value taken by the parameter for the next horizon steps
    pal_koff,#array with the value taken by the parameter for the next horizon steps
    initialization:bool,#if it is the first time point
):
    out_increments = []
    tau_memory = int(round(contact_memory_time[0]))
    for tip,is_contact_tip in zip(tip_keys,is_contact_flags):
        prog_tip = prog_data.get(tip,[])
        if len(prog_tip) <= 1 or (is_contact_tip and len(prog_tip) <= tau_memory+1):
            if program_type == PROGRAM_GENNORM:
                prog_tip = gennorm.rvs(gennorm_beta,gennorm_loc*dt,gennorm_scale*(dt**(1/gennorm_beta)))
            elif program_type == PROGRAM_PALAVALLI:
                if len(prog_tip) == 1:
                    initial_phase = int(np.sign(prog_tip[0]) == -1)
                else:
                    if initialization:
                        initial_phase = np.random.randint(2)
                    else:
                        initial_phase = 0#branch birth is growth
                    prog_tip = palavalli_numba(horizon,dt,pal_kon,pal_koff,initial_phase,pal_von,pal_voff)
                

            elif program_type == PROGRAM_ARFIMA:
                if len(prog_tip) == 0:
                    prog_tip = arfima(arfima_d,arfima_kappa_eps,arfima_loc_eps,arfima_scale_eps,horizon,dt,arfima_p_mb,arfima_d_mb)
                    # if np.any(np.isnan(prog_tip)):
                    #     print(prog_tip)
                    #     0
                    # print(prog_tip[0])
        if is_contact_tip:
            contact_retraction = np.random.normal(contact_mean_retraction[0],contact_std_retraction[0])
            post_contact_incr = np.linspace(0,contact_retraction,tau_memory+1)[1:]
            if prog_tip.shape[0] > post_contact_incr.shape[0] :
                prog_tip[:post_contact_incr.shape[0]] = post_contact_incr
            else:
                prog_tip = post_contact_incr
        #get_next_incr
        out_increments.append(prog_tip[0])
        prog_tip = prog_tip[1:]

        #update program dict
        prog_data[tip] = prog_tip

        #contact penalty
        # if is_contact_tip:
        #     out_increments[-1] -= contact_penality[0]
    return(to_numba_float_list(out_increments),prog_data)



def arfima(
        d : float,
        arfima_kappa_eps,#arrayx with the shape (horizon,), the value of the parameter for the next horizon time steps
        arfima_loc_eps,#arrayx with the shape (horizon,), the value of the parameter for the next horizon time steps
        arfima_scale_eps,#arrayx with the shape (horizon,), the value of the parameter for the next horizon time steps
        horizon : int,
        dt : float,
        arfima_p_mb:float,
        arfima_d_mb:float,
):
    if np.random.random() < arfima_p_mb:
        d = arfima_d_mb
    burnin = max(1024, horizon // 2)
    N = horizon + burnin  # simulate longer, then trim

    # w = special.poch(d, np.arange(N)) / special.gamma(np.arange(N) + 1)
    temp_n = np.arange(N)
    if abs(d) < 1e-8:
        w = np.zeros_like(temp_n)
        w[0] = 1
    else:
        # w = np.exp(special.gammaln(d + temp_n) - special.gammaln(d) - special.gammaln(temp_n + 1))
        w = np.empty(N, dtype=float)
        w[0] = 1.0
        for j in range(1, N):
            w[j] = w[j-1] * (j - 1 + d) / j

    eps = np.zeros((N,))
    # print(arfima_kappa_eps[0])
    eps[:burnin] = laplace_asymmetric.rvs(arfima_kappa_eps[0],loc = arfima_loc_eps[0]*dt,scale = arfima_scale_eps[0]*dt,size = burnin)
    eps[burnin:burnin+arfima_kappa_eps.shape[0]] =  laplace_asymmetric.rvs(arfima_kappa_eps,loc = arfima_loc_eps*dt,scale = arfima_scale_eps*dt)

    L = 1 << ((2*N - 2).bit_length())

    E = np.fft.rfft(eps, n=L)
    W = np.fft.rfft(w,   n=L)
    y_full = np.fft.irfft(E * W, n=L)[:N]  # keep causal part

    # trim burn-in
    y = y_full[burnin:]

    return(y)


@njit(cache=True)
def palavalli_numba(
    horizon: int,
    dt: float,
    pal_kon: np.ndarray,
    pal_koff: np.ndarray,
    initial_phase: int,
    pal_von: np.ndarray,
    pal_voff: np.ndarray,
) -> np.ndarray:
    """
    Numba-accelerated two-state (on/off) tip-dynamics simulator with time-scheduled rates/velocities.
    - state 0: growth, exponential duration with rate pal_koff[idx], velocity +pal_von[idx]
    - state 1: shrink, exponential duration with rate pal_kon[idx], velocity -pal_voff[idx]
    Parameters at time t use the nearest index round(t/dt).
    Returns the tip position sampled on the regular grid new_time = np.arange(horizon)*dt.
    """
    nT = horizon
    prog_tip = np.zeros(nT, dtype=np.float64)

    # Time grid is implicit: t_i = i*dt
    # Fill progressively by simulating event intervals [t_prev, t_next)
    t_prev = 0.0
    x_prev = 0.0
    i = 0  # grid index to fill next
    state = 0 if initial_phase == 0 else 1

    # small floor for rates to avoid division by zero
    rate_floor = 1e-12

    while i < nT:
        # choose parameters from the nearest grid index to current time
        idx = int(round(t_prev / dt))
        if idx < 0:
            idx = 0
        elif idx >= nT:
            idx = nT - 1

        if state == 0:  # growth phase
            rate = pal_koff[idx]
            if rate < rate_floor:
                rate = rate_floor
            scale = 1.0 / rate
            vel = pal_von[idx]
            next_state = 1
        else:  # shrink phase
            rate = pal_kon[idx]
            if rate < rate_floor:
                rate = rate_floor
            scale = 1.0 / rate
            vel = -pal_voff[idx]
            next_state = 0

        # draw an exponential duration with current scale
        duration = np.random.exponential(scale)
        t_next = t_prev + duration
        x_next = x_prev + vel * duration

        # fill all regular-grid samples in [t_prev, t_next)
        # linear interpolation between (t_prev, x_prev) and (t_next, x_next)
        while i < nT:
            t = i * dt
            if t < t_next:
                # interpolate
                denom = t_next - t_prev
                if denom > 0.0:
                    frac = (t - t_prev) / denom
                    prog_tip[i] = x_prev + frac * (x_next - x_prev)
                else:
                    # extremely rare: zero duration: just copy current value
                    prog_tip[i] = x_prev
                i += 1
            else:
                break

        # advance to next event
        t_prev = t_next
        x_prev = x_next
        state = next_state

    return(prog_tip[1:] - prog_tip[:-1])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    n_trajs = 100
    horizon = 1000
    all_trajs = []
    for i in range(n_trajs):
        all_trajs.append(np.cumsum(palavalli_numba(horizon,1,0.5*np.ones(horizon),1*np.ones(horizon),np.random.randint(2),0.7*np.ones(horizon),0.5*np.ones(horizon))))
    t = np.arange(horizon)
    all_trajs = np.array(all_trajs)
    slope, intercept, r, p, se = linregress(np.log(t[1:]),np.log(np.mean((all_trajs-np.mean(all_trajs,axis = 0))**2,axis = 0)))
    print(slope)
    for traj in all_trajs:
        plt.plot(traj)
    plt.show()