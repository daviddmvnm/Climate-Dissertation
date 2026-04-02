Claude Code Instructions: SMM Calibration — Alpha Parameters Only

Context
We have a finite-horizon dynamic Markov game modelling climate cooperation between 4 geopolitical blocs (EU, US, CN, RoW). The model is implemented in Python as functions taking parameters as inputs and returning equilibrium outcomes via backward induction and Monte Carlo simulation.
We are implementing SMM to anchor the four unobservable cardinal scaling parameters (alphas) to real-world 2023 data. Read the existing Python file in full before writing any code. Do not assume function signatures or parameter names — inspect everything first.

The Logic of What We're Doing
The model has two types of parameters. The first type has empirical anchors — discount factors are grounded in observed bond markets, policy cycles, and Nordhaus (2007); the rationality parameter λ is grounded in the experimental literature. These stay fixed. The second type has no empirical anchor — the alpha scaling parameters determine how strongly each payoff channel (costs, damages, pressure, benefits) enters the strategic calculus, and no data directly identifies their cardinal magnitude. These are what SMM disciplines.
This is the core methodological claim: SMM identifies what the data can identify and leaves fixed what it cannot.

Parameters — Fixed vs Calibrated

Fix these at original calibrated values. Do not move them:
pythonlambda_  = 1.54
delta_EU = 0.85
delta_US = 0.75
delta_CN = 0.80
delta_RoW = 0.65

Calibrate these via SMM — genuinely unobservable:
pythonalpha_c  # transition cost scaling
alpha_d  # climate damage scaling
alpha_p  # political pressure scaling
alpha_b  # coordination benefit scaling

The 3 Target Moments
pythonmoments_target = np.array([
    2.0,    # EU/US adoption ratio
            # EU renewable share 45.3% / US 22.7%
            # Sources: Eurostat (2023); EIA Electric Power Monthly (2024)

    1.56,   # EU/CN adoption ratio
            # EU renewable share 45.3% / CN 29%
            # Sources: Eurostat (2023); IEA China energy profile (2024)

    0.305   # Weighted coalition depth W in 2023
            # IRENA (2024) renewable capacity shares weighted by
            # bloc influence weights from the model:
            # EU(0.20 × 0.453) + US(0.25 × 0.227)
            # + CN(0.37 × 0.290) + RoW(0.18 × 0.280)
])

Implementation
Step 1 — Moment extraction function
pythondef compute_moments(alpha_params):
    alpha_c, alpha_d, alpha_p, alpha_b = alpha_params
    
    # Call existing model solver with fixed discount factors
    # and variable alphas — use whatever the actual function
    # signature is after reading the file
    results = solve_model(
        lambda_=1.54,
        delta_EU=0.85, delta_US=0.75,
        delta_CN=0.80, delta_RoW=0.65,
        alpha_c=alpha_c, alpha_d=alpha_d,
        alpha_p=alpha_p, alpha_b=alpha_b
    )
    
    # Extract period-1 equilibrium probabilities
    sigma_EU = results['period1_sigma']['EU']
    sigma_US = results['period1_sigma']['US']
    sigma_CN = max(results['period1_sigma']['CN'], 0.001)  # floor to avoid blowup
    sigma_RoW = results['period1_sigma']['RoW']
    
    # Compute weighted W at period 1
    W_model = (0.20 * sigma_EU + 0.25 * sigma_US +
               0.37 * sigma_CN + 0.18 * sigma_RoW)
    
    return np.array([
        sigma_EU / sigma_US,
        sigma_EU / sigma_CN,
        W_model
    ])

Step 2 — Objective function
pythondef smm_objective(alpha_params):
    alpha_c, alpha_d, alpha_p, alpha_b = alpha_params
    
    # Hard bounds
    if not (0.5 <= alpha_c <= 6.0): return 1e6
    if not (0.05 <= alpha_d <= 1.0): return 1e6
    if not (0.1 <= alpha_p <= 5.0): return 1e6
    if not (0.5 <= alpha_b <= 6.0): return 1e6
    
    moments_model = compute_moments(alpha_params)
    diff = moments_target - moments_model
    return diff @ diff

Step 3 — Optimisation
Run from all 3 original scenarios as starting points. This is methodologically meaningful — you are asking which scenario the data most supports.
pythonstarting_points = [
    [3.5, 0.30, 0.5, 3.0],  # Scenario A
    [2.0, 0.15, 2.5, 1.0],  # Scenario B
    [2.0, 0.30, 0.5, 1.0],  # Scenario C
]

results_all = []
for x0 in starting_points:
    result = minimize(
        smm_objective, x0,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-4,
                 'fatol': 1e-4, 'disp': True}
    )
    results_all.append(result)
    print(f"Start {x0} → {result.x}, obj={result.fun:.6f}")


Step 4 — Do NOT constrain baseline success rate
After finding SMM estimates, run Monte Carlo with 1000 draws and report whatever baseline success rate emerges. Do not target 50% — let it be a finding. Three possible outcomes each tell a story:

Near 50% → data validates original grid search
High (70-90%) → world is closer to coordination than assumed, strengthens policy optimism
Low (20-30%) → coordination harder than assumed, different policy implications

All three are writable. Report whatever comes out honestly.

Step 5 — Output tables

Table 1 — Moment Fit
MomentDataPre-SMMPost-SMMEU/US adoption ratio2.00[compute][compute]EU/CN adoption ratio1.56[compute][compute]Weighted W0.305[compute][compute]

Table 2 — SMM Alpha Estimates vs Original Scenarios
ParameterScen AScen BScen CSMM Estimateα_c3.52.02.0[result]α_d0.300.150.30[result]α_p0.52.50.5[result]α_b3.01.01.0[result]

Table 3 — Baseline Diagnostics Under SMM
MetricOriginalSMMCoordination success rate~0.50[result]Mean final W[original][result]
EU period-1 σ[original][result]CN period-1 σ[original][result]

smm_calibration.py — full SMM implementation
smm_output.txt — all printed results
smm_tables.py — clean table generation for dissertation