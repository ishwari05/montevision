"""
Black Swan / Extreme Market Event Engine
Generates stochastic systemic shocks modeled as clustered Markov chains.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_crash_events(days: int, n_simulations: int, config) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates rare tail-risk crash events using a hybrid Poisson-Markov approach.
    Initial crashes are pre-sampled via Poisson, while cascading crashes trigger 
    probabilistically during crisis windows.
    """
    logger.info("Injecting realistic Black Swan events (Poisson + Controlled Clustering)...")
    
    lam = getattr(config, 'BLACK_SWAN_LAMBDA', 1.0)
    max_crashes = getattr(config, 'MAX_CRASHES_PER_SIM', 3)
    crisis_prob = getattr(config, 'CRISIS_PROB_DAILY', 0.005)
    vol_spike = getattr(config, 'VOLATILITY_SPIKE_MULTIPLIER', 2.0)
    min_drop, max_drop = getattr(config, 'BLACK_SWAN_MAGNITUDE_RANGE', (-0.35, -0.15))
    min_dur, max_dur = getattr(config, 'CRISIS_DURATION_RANGE', (10, 30))
    
    crash_multipliers = np.ones((days, n_simulations))
    vol_multipliers = np.ones((days, n_simulations))
    
    # 1. Pre-sample "Initial" crashes using Poisson distribution
    num_initial_crashes = np.random.poisson(lam, n_simulations)
    # Cap the initial crashes by both the logic limit and the physics of the timeline
    num_initial_crashes = np.minimum(num_initial_crashes, max_crashes)
    num_initial_crashes = np.minimum(num_initial_crashes, days)
    
    # Track crashes per simulation
    crashes_count = np.zeros(n_simulations, dtype=int)
    crisis_days_remaining = np.zeros(n_simulations, dtype=int)
    
    # Assign initial crash timesteps randomly
    for i in range(n_simulations):
        if num_initial_crashes[i] > 0:
            crash_days = np.random.choice(days, num_initial_crashes[i], replace=False)
            # Initialize with random drops
            drops = np.random.uniform(min_drop, max_drop, num_initial_crashes[i])
            crash_multipliers[crash_days, i] = 1.0 + drops
            # We don't increment crashes_count here yet, we'll do it in the loop to handle clustering logic
            
    total_triggered_crashes = 0
    sims_with_crash = np.zeros(n_simulations, dtype=bool)

    # 2. Sequential simulation to handle Clustering and Volatility spikes
    for t in range(days):
        # Identify simulations currently in a crisis
        in_crisis = (crisis_days_remaining > 0)
        
        # Check if we can still crash
        can_crash_more = (crashes_count < max_crashes)
        
        # Identify simulations that have a pre-sampled crash at this timestep
        # AND are actually allowed to crash now
        pre_sampled_crash = (crash_multipliers[t, :] < 1.0) & can_crash_more
        
        # Check for *cascading* crashes only if in crisis and haven't hit max limit
        cascade_roll = np.random.random(n_simulations)
        # Cascading crashes are strictly state-dependent
        cascade_crash = (in_crisis & can_crash_more & (cascade_roll < crisis_prob) & ~pre_sampled_crash)
        
        # Combine crashes
        actual_crash_this_step = pre_sampled_crash | cascade_crash
        
        # Suppress pre-sampled crashes that were over the limit (cleanup)
        # If there was a pre-sampled crash but can_crash_more was false, we effectively ignore it
        if np.any((crash_multipliers[t, :] < 1.0) & ~can_crash_more):
            crash_multipliers[t, (crash_multipliers[t, :] < 1.0) & ~can_crash_more] = 1.0
        
        if np.any(actual_crash_this_step):
            num_new = np.sum(actual_crash_this_step)
            total_triggered_crashes += num_new
            sims_with_crash[actual_crash_this_step] = True
            crashes_count[actual_crash_this_step] += 1
            
            # Apply drop for cascade crashes
            if np.any(cascade_crash):
                num_cascade = np.sum(cascade_crash)
                drops = np.random.uniform(min_drop, max_drop, num_cascade)
                crash_multipliers[t, cascade_crash] = 1.0 + drops
                
            # (Re)set crisis duration
            durations = np.random.randint(min_dur, max_dur + 1, num_new)
            crisis_days_remaining[actual_crash_this_step] = durations
            
        # Apply volatility spikes
        vol_multipliers[t, crisis_days_remaining > 0] = vol_spike
        
        # Countdown crisis
        crisis_days_remaining[in_crisis] -= 1

    # Cleanup: If a pre-sampled crash didn't trigger (hit max cap), reset it
    # (Though in this logic, we process pre-sampled as part of actual_crash_this_step)
    
    avg_crashes = total_triggered_crashes / n_simulations
    crash_percent = (np.sum(sims_with_crash) / n_simulations) * 100
    
    print("\n" + "="*40)
    print("📉 BLACK SWAN REALISM SUMMARY")
    print("-" * 40)
    print(f"Avg crashes per simulation: {avg_crashes:.2f}")
    print(f"Simulations with ≥1 crash: {crash_percent:.1f}%")
    print("="*40 + "\n")
    
    return crash_multipliers, vol_multipliers
