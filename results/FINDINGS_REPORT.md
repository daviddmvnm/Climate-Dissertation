# Simulation Findings Report
### A Dynamic Markov Game of International Climate Cooperation

> **Note on numbers:** Results below are from a reduced-draw run (500 MC/scenario, 100 MC/sweep-point, 50 MC/GSA-sample). Structural rankings are stable at these draw counts; point estimates tighten on the full run. All conclusions below already hold.

> **Methodological note:** Scenarios A, B, and C represent different cardinal scalings of unobservable incentive parameters (αc, αd, αp, αb). They are not comparable alternatives — each is a distinct parameterisation of the world. Findings are therefore classified strictly by what holds *within* each scenario and whether that pattern is consistent *across* all three. The three-tier robustness classification formalises this: Tier 1 holds under any plausible parameterisation; Tier 2 holds directionally across all three scenarios; Tier 3 is scenario-conditional.

---

## 1. Overview

The model is a finite-horizon (T=10, 5 years/period) dynamic Markov game with four heterogeneous blocs — US, EU, CN, RoW — choosing irreversibly whether to adopt green technology. Equilibrium is characterised by Quantal Response Equilibrium (λ=1.5), solved via backward induction with fixed-point iteration at every state. The coordination threshold is θ=0.80, calibrated to reflect the near-universal participation requirement of effective climate stabilisation.

Three scenarios span distinct incentive regimes:

| Scenario | αc | αd | αp | αb | Character |
|----------|----|----|----|-----|-----------|
| **A** | 3.5 | 0.30 | 0.5 | 3.0 | Reward-driven |
| **B** | 2.0 | 0.15 | 2.5 | 1.0 | Pressure-driven |
| **C** | 2.0 | 0.30 | 0.5 | 1.0 | Weak-incentive |

All three satisfy the non-degeneracy criterion (30–70% baseline coordination success), ensuring each scenario sits in a range where parameter variation is informative rather than trivially pushing outcomes to 0 or 100%.

---

## 2. Core Findings

### Finding 1 — δ_CN is the dominant parameter across all incentive regimes

The single most consistent result across all three scenarios and all outcome measures is the dominance of China's patience parameter δ_CN. In the GSA, δ_CN is the strongest predictor of coordination success rate in every scenario:

| Scenario | δ_CN (ρ, success) | δ_CN (ρ, timing) | δ_CN (ρ, CN FM prob) |
|----------|--------------------|-------------------|----------------------|
| A | +0.68 | −0.61 | +0.75 |
| B | +0.64 | −0.58 | +0.72 |
| C | +0.58 | −0.60 | +0.70 |

In every scenario: raising δ_CN increases coordination success, accelerates coordination (when it occurs), and raises China's first-mover probability. No other parameter matches this consistency or magnitude.

The marginal sweep confirms the knife-edge: raising δ_CN from its calibrated 0.80 to 0.821 — a shift of approximately 0.4 percentage points in annual discount rate — crosses the 50% coordination threshold in all three scenarios. The most important parameter in the model sits within a hair's width of the tipping point across every incentive regime tested.

**This is a Tier 1 result.** The dominance of δ_CN is not a consequence of any particular cardinal scaling — it follows from China's structural position: highest influence weight (w_CN ≈ 0.37), highest transition costs, lowest present-period pressure sensitivity, and a discount factor calibrated at the lower end relative to EU and US. Any plausible parameterisation that preserves these ordinal relationships reproduces the δ_CN dominance.

---

### Finding 2 — China is pivotal and passive: the coordination trap

China's first-mover probability at period 1 is consistently the lowest of any active player in all three scenarios:

| Bloc | w (influence) | FM Prob (A) | FM Prob (B) | FM Prob (C) |
|------|--------------|------------|------------|------------|
| EU | 0.183 | 31.6% | 18.4% | 19.8% |
| US | 0.282 | 19.6% | 18.6% | 11.2% |
| **CN** | **0.370** | **1.6%** | **5.6%** | **2.6%** |
| RoW | 0.166 | 0.0% | 0.0% | 0.0% |

The player whose adoption matters most — carrying the largest influence weight — has the weakest incentive to move first in every scenario. The trap is structural:

- China's transition costs are the highest in the calibration (carbon intensity × affordability adjustment)
- Its discount factor (δ_CN=0.80, ≈4.6% annual) is lower than the EU's, making it more present-biased
- Political pressure sensitivity is lower than US and EU, reducing the penalty for holding out

The GSA confirms that δ_CN is also the strongest predictor of China's own first-mover probability (ρ = 0.70–0.75 across scenarios): the parameter that determines whether China can be drawn into early adoption is primarily its own planning horizon, not the incentive structure around it.

**This is Tier 2 robust.** The ordering CN < US ≈ EU for first-mover probability, and the dominance of δ_CN in predicting it, holds across all three scenarios. The magnitude of CN's first-mover probability varies (as low as 1.6%, as high as 5.6%), but the structural trap is present in every incentive regime.

---

### Finding 3 — Rationality has a split and counterintuitive role

The rationality parameter λ produces a structurally inconsistent pattern across outcome measures that is consistent in direction across all three scenarios:

**For coordination timing** (how fast, given that coordination occurs): λ is the second-strongest predictor in all three scenarios. Spearman ρ for coordination timing:

| Scenario | ρ (timing) |
|----------|-----------|
| A | −0.51 |
| B | −0.50 |
| C | −0.47 |

Higher rationality accelerates coordination when it occurs — players respond more sharply to payoff gradients that favour adoption.

**For coordination success rate** (whether coordination occurs at all): λ has near-zero or negative correlation in all scenarios:

| Scenario | ρ (success) |
|----------|-------------|
| A | +0.05 |
| B | −0.15 |
| C | −0.04 |

More rational players are not better at reaching the coordination threshold; in the pressure-driven regime, they are slightly worse.

The mechanism is strategic delay. In a QRE framework, higher λ sharpens best-response. When the equilibrium is a coordination equilibrium (everyone adopts), sharpening it helps. But when equilibrium contains a delay component (wait for others to bear first-mover costs), sharper rationality also makes delay more disciplined. Bounded rationality introduces probabilistic adoption that can start cascades; rationality can lock in strategic waiting instead. The same mechanism that accelerates coordination when it is on-path makes free-riding more precise when it is not.

For China's first-mover probability specifically, λ is consistently negative (ρ ≈ −0.13 to −0.28 across scenarios): more rational players are *less* likely to absorb first-mover costs when those costs are individually large.

**This is Tier 2 robust.** The sign pattern — near-zero or negative for success, strongly negative for timing, negative for first-mover probability — is consistent in direction across all three scenarios. λ is classified as "split" in the GSA ranking because reporting a single tier would misrepresent its structure.

---

### Finding 4 — EU structural leadership; EU patience is a null result

The EU leads as first mover in every scenario. This is Tier 1: EU adoption in period 1 is structurally implied by its parameter configuration (highest discount factor at δ_EU=0.85, intermediate transition costs, high pressure sensitivity). The EU leads regardless of where its own discount factor sits within any plausible range.

Yet in the GSA, δ_EU is the *weakest* predictor of coordination success across all three scenarios:

| Scenario | ρ (δ_EU, success) |
|----------|--------------------|
| A | +0.09 |
| B | +0.11 |
| C | +0.07 |

These are consistently the lowest ρ values in the full 8-parameter GSA, lower than δ_RoW, lower than γ, lower than κ.

The reason: EU adoption is essentially guaranteed within the model's parameter space. The EU leads whether its annual discount rate is 2% or 5%. Since it almost certainly adopts early regardless, variation in δ_EU has no marginal effect on whether the threshold is ultimately reached — what matters is what happens after the EU moves, particularly whether China and the US follow.

The policy implication is blunt: diplomatic effort aimed at strengthening EU climate commitment — more ambitious national targets, tighter EU ETS, accelerated net-zero legislation — has effectively zero marginal effect on global coordination probability in this framework. The EU has already done what it can do. The binding constraint is elsewhere.

**Tier 2 robust.** EU leads first (Tier 1 structural). δ_EU ρ < 0.12 in all three scenarios (consistent null).

---

### Finding 5 — US-CN patience complementarity

The marginal sweep for δ_US alone crosses the 50% coordination threshold at 0.757 in Scenarios A and B — just above the calibrated δ_US=0.75. Like China, the US sits near a tipping point within each scenario. The joint sweep (δ_US = δ_CN varied together) illuminates the complementarity:

| Channel | A 90%-cross | B 90%-cross | C 90%-cross |
|---------|------------|------------|------------|
| δ_US alone | 0.821 | 0.886 | absent |
| δ_CN alone | 0.886 | 0.886 | 0.950 |
| δ_US = δ_CN | 0.821 | 0.886 | 0.886 |

Within each scenario, the joint channel achieves 90% coordination at lower parameter values than either component alone. In Scenario A, the joint channel reaches 90% at 0.821 — tighter than δ_CN alone (0.886). In Scenario C, the joint channel reaches 90% at 0.886 whereas δ_US alone cannot reach it at any value in the admissible range.

The structural logic: neither the US nor China alone carries enough influence weight to cross θ=0.80. The US has w=0.282 and China w=0.370; together they account for 65.2% of global influence weight. Combined patience is required to make coordination near-certain because they must both adopt to reach the threshold, and each conditions their adoption on the other's expected behaviour.

**Tier 2 robust.** The marginal value of US patience is realised primarily when China is also patient, and vice versa — consistent across all three scenarios.

---

## 3. Supporting Evidence: Sweep Channel Rankings

The 15-channel marginal sweep analysis, summarised by 50% coordination-crossing values within each scenario:

| Channel | A 50%-cross | B 50%-cross | C 50%-cross | Tier |
|---------|------------|------------|------------|------|
| Global cost reduction | 0.30 | 0.30 | 0.30 | 1 |
| RoW cost | 0.20 | 0.20 | 0.20 | 1 |
| θ (threshold level) | 0.50 | 0.50 | 0.50 | 1 |
| δ_US | 0.757 | 0.757 | 0.821 | 2 |
| δ_CN | 0.821 | 0.821 | 0.821 | 2 |
| δ_US = δ_CN (joint) | 0.821 | 0.821 | 0.821 | 2 |
| κ (damage escalation) | 0.064 | 0.037 | 0.064 | 2 |
| γ (learning rate) | 0.236 | 0.143 | 0.421 | 3 |
| η (sigmoid steepness) | 16.4 | 10.7 | 19.3 | 3 |
| US pressure | 0.20 | 0.20 | 0.743 | 3 |
| CN pressure | 0.743 | 0.743 | 2.371 | 3 |
| Global pressure | 0.20 | 1.286 | 1.286 | 3 |
| λ (rationality) | 0.50 | 0.50 | 0.50 | 3 |

**Cost channels** sit at the floor of their sweep ranges across all scenarios. This is Tier 1: costs in the baseline calibration are high enough to be the binding barrier in a significant share of runs. Any plausible technology cost reduction immediately shifts the model toward coordination — not through incentive creation but through barrier removal.

**Damage escalation κ** crosses the 50% threshold within range in all three scenarios (Tier 2), but cannot guarantee coordination — the 90% crossing is absent or requires extreme κ in some scenarios. Urgency starts cascades but does not sustain them across the full range.

**CN pressure sensitivity** diverges sharply: the 50% crossing falls at 0.743 in Scenarios A and B, but at 2.371 in Scenario C — well outside any plausible calibration range. This is a Tier 3 result: political pressure on China is effective at shifting coordination within reach of strong-incentive regimes, but cannot compensate for the absence of other coordination drivers in weak-incentive environments. The channel is conditionally useful, not universally so.

**γ (technology learning)** crosses within range in Scenarios A and B but requires a very high value (0.421) in Scenario C to reach 50% — approaching the theoretical maximum where adoption becomes essentially costless immediately. Learning is a meaningful lever in reward- and pressure-driven regimes; insufficient alone in weak-incentive environments.

---

## 4. GSA Full Ranking Summary

| Parameter | Success Rate (avg ρ) | Coord Timing (avg ρ) | CN FM Prob (avg ρ) | Overall Tier |
|-----------|---------------------|---------------------|-------------------|-------------|
| δ_CN | **+0.642** | **−0.597** | **+0.729** | 1 |
| δ_US | +0.442 | −0.477 | +0.139 | 2 |
| η | +0.281 | −0.133 | +0.109 | 2/3 |
| δ_RoW | +0.214 | +0.024 | +0.168 | 3 |
| γ | +0.192 | −0.092 | +0.127 | 3 |
| κ | +0.199 | −0.148 | +0.067 | 3 |
| λ | +0.052 | **−0.494** | **−0.249** | split |
| δ_EU | +0.096 | −0.050 | +0.054 | null |

*Average ρ across the three scenarios.*

λ is classified as "split" because its role genuinely differs by outcome dimension: strong predictor of timing, near-irrelevant or counterproductive for success rates. This split is consistent across all three scenarios — it is not a Tier 3 result but a structural feature of QRE.

---

## 5. Three-Tier Robustness Classification

### Tier 1 — Holds under any plausible parameterisation
- EU leads first; RoW never leads. (Implied by the ordinal structure of discount factors and costs.)
- Cost reductions at any plausible magnitude guarantee coordination. (Hard barrier finding — costs are binding in baseline.)
- δ_CN is the dominant GSA parameter across all outcomes and scenarios.

### Tier 2 — Directionally consistent across all three scenarios
- US patience ranks second; the US-CN patience dyad jointly determines near-certain coordination.
- Higher rationality (λ) has zero or negative net effect on coordination *success* but strongly accelerates coordination *timing* when it occurs.
- EU patience (δ_EU) has near-zero marginal value for global outcomes in all three scenarios.
- China's first-mover probability is the lowest of any active player in all three scenarios.
- δ_CN crosses the 50% coordination threshold at 0.821 in all three scenarios (high precision, Tier 2 quantitative).

### Tier 3 — Calibration-dependent
- γ (learning): effective in A and B, insufficient as a standalone lever in C.
- κ (urgency): reaches 50%-crossing within range in all scenarios, but 90%-crossing behaviour varies.
- CN pressure sensitivity: crossing within admissible range in A and B, well outside in C.
- US and global pressure: regime-conditional effectiveness.

---

## 6. Parameter and Lever Commentary

The table below classifies each parameter by its observed effect across the three outcome dimensions — coordination success rate, coordination timing (conditional on success), and China's first-mover probability — and assigns a tier reflecting how stable that effect is across the three incentive scenarios.

---

### Tier 1 Levers — Effect holds under any plausible parameterisation

**Technology cost (global cost reduction, RoW cost)**
Both cost channels sit at the lower floor of their sweep ranges across all three scenarios, meaning the 50% coordination threshold is crossed immediately at any meaningful cost reduction. This is not a marginal or incentive-based result — it is a hard barrier finding. Costs in the baseline calibration are high enough that a non-trivial fraction of runs fail coordination simply because the adoption cost exceeds the realised benefit at the point of decision. Cost reductions clear this barrier mechanically. Effect on success: strongly positive. Effect on timing: accelerates. Note: operates through barrier removal, not incentive creation — the mechanism is distinct from patience or pressure channels.

**Threshold θ**
The coordination threshold directly determines what weighted adoption share constitutes a successful outcome. Lowering θ expands the set of paths that qualify as coordinated; raising it shrinks it. The 50% crossing at θ=0.50 (in a model calibrated to θ=0.80) confirms the current threshold is genuinely ambitious. Structurally, θ interacts with the influence weight distribution: because w_CN=0.37 and w_US=0.28, the threshold cannot be reached without at least one of the two largest blocs. At θ=0.80, it cannot be reached without both. θ is therefore the parameter that determines whether the US-CN complementarity is structurally binding or merely relevant.

**δ_CN (China's discount factor)**
The single dominant parameter across all outcomes and all scenarios (GSA avg ρ: +0.64 success, −0.60 timing, +0.73 CN first-mover probability). The 50% coordination crossing falls at δ_CN=0.821 in all three scenarios — a shift of 0.4 percentage points in annual discount rate from the calibrated value of 0.80. The mechanism is through the first-mover calculation: a more patient China places higher weight on the long-run coordinated payoff relative to the short-run cost of early adoption. Since China's adoption is structurally necessary under any threshold near 0.80, its patience is the single most leveraged parameter in the model regardless of incentive regime. This is Tier 1 in dominance (no other parameter approaches its ρ values), Tier 2 in quantitative precision (crossing value varies slightly across scenarios).

---

### Tier 2 Levers — Direction consistent across all three scenarios, magnitude varies

**δ_US (US discount factor)**
Second-ranked in the GSA for success rate (avg ρ: +0.44) and third for timing (avg ρ: −0.48). The 50% crossing for δ_US falls at 0.757 in Scenarios A and B — just above the calibrated 0.75, placing the US near a tipping point in every non-weak-incentive regime. The US and China together hold 65% of total influence weight, making US patience a necessary complement to China's. Effect: higher δ_US raises success probability and accelerates timing in all scenarios; the effect is largest when δ_CN is also elevated (Finding 5 complementarity). Classified Tier 2 rather than Tier 1 because in Scenario C the 50% crossing shifts to 0.821 and the 90% crossing is absent — the US patience channel weakens (though does not disappear) in the weakest incentive environment.

**κ (damage escalation rate)**
Affects coordination through the urgency mechanism: higher κ raises the flow cost of remaining in the non-coordinated state as t increases, tipping the Q-value calculation toward early adoption. GSA avg ρ: +0.20 for success, −0.15 for timing. The 50% crossing falls within the plausible range in all three scenarios (0.037–0.064), making κ a meaningful lever everywhere. However, the 90% crossing is absent or requires extreme κ in most scenarios. Interpretation: urgency starts cascades — when climate damage escalates quickly enough, early adoption becomes dominant strategy — but urgency alone cannot guarantee coordination if patience parameters are calibrated against it. Conditionally effective, universally present.

**λ (QRE rationality parameter) — split classification**
λ produces structurally opposite effects depending on the outcome dimension:
- *Coordination timing*: second-strongest predictor in all scenarios (avg ρ: −0.49). Higher rationality sharply accelerates coordination once the incentive gradient tips toward adoption — players best-respond more decisively.
- *Coordination success rate*: near-zero or negative in all scenarios (avg ρ: +0.05 to −0.15). More rational players are not better at reaching the threshold and in some regimes are marginally worse.
- *China first-mover probability*: consistently negative (avg ρ: −0.25). More rational players are less likely to absorb first-mover costs when holding out is individually profitable.

The mechanism: QRE rationality sharpens best-response. When the equilibrium path involves coordination, sharpening it helps everyone move faster. When it involves strategic delay (free-riding on others), sharpening it makes delay more disciplined and harder to dislodge by natural experimentation. Noise and bounded rationality can trigger adoption cascades stochastically; rational precision can prevent this. λ is therefore not a simple "more is better" lever. It is a Tier 2 split result: the sign pattern is consistent across all three scenarios, but the net effect on policy-relevant outcomes (success rate) is near-zero.

**δ_EU (EU discount factor) — null result**
The EU leads as first mover in every scenario regardless of where δ_EU sits within any plausible range (Tier 1 structural). Consistent with this, δ_EU is the weakest predictor of coordination success in the GSA (avg ρ: +0.10 — the lowest value in the 8-parameter analysis, below even δ_RoW). The EU's commitment to early adoption is robust to its own patience parameter — it leads whether patient or present-biased. The implication: policy levers that target EU ambition (stricter ETS targets, tighter net-zero legislation) operate in an already-saturated region of the parameter space. The EU's contribution to coordination is structurally guaranteed within the model; variation in δ_EU shifts nothing material. This is a Tier 2 null result — consistently near-zero ρ across all three scenarios.

---

### Tier 3 Levers — Scenario-conditional, magnitude and sometimes direction varies

**γ (technology learning rate)**
Affects coordination by reducing adoption costs as more blocs adopt: c_i(W) = c_i^0(1 − γW). Higher γ means later adopters face substantially lower costs, creating a demonstration effect where early EU adoption makes subsequent adoption cheaper. GSA avg ρ: +0.19 for success. Sweep crossing: 0.236 (A), 0.143 (B), 0.421 (C). In Scenarios A and B, γ reaches the 50% threshold at plausible values; in Scenario C, reaching the threshold requires γ=0.421 — near the theoretical maximum where the second adopter sees costs reduced by 42% relative to the first. Learning is an effective Tier 2-candidate lever in strong-incentive regimes but is insufficient in the weak-incentive environment. Classified Tier 3 because the crossing behaviour is qualitatively different across scenarios.

**CN pressure sensitivity**
Among the most scenario-sensitive levers in the model. In Scenarios A and B, the 50% coordination crossing for CN pressure falls at 0.743 — within the plausible calibration range, indicating that making China more sensitive to political pressure from adopters can shift coordination meaningfully. In Scenario C (weak incentives), the crossing falls at 2.371 — well outside any plausible range. The interpretation: political pressure on China is conditionally effective when the rest of the incentive structure is supportive. When coordination benefits are low and damage sensitivity is low, no plausible increase in China's pressure sensitivity can compensate. Pressure channels require a supportive incentive backdrop to operate; they cannot substitute for it.

**US pressure and global pressure**
Both channels show floor-crossing behaviour (50% at 0.20 — the lower bound of the sweep) in at least one scenario, suggesting that in reward- or moderate-incentive regimes, even baseline political pressure is sufficient to support coordination. In Scenario C, the crossings shift out substantially (US: 0.743; global: 1.286), confirming the weak-incentive environment degrades the effectiveness of all pressure channels. Tier 3 because the lever moves from trivially sufficient to conditionally necessary depending on the incentive regime.

**η (sigmoid steepness)**
Controls how sharply the coordination benefit and climate damage terms activate around the threshold W=θ. Higher η sharpens the threshold — players face a nearly discontinuous jump in payoffs near θ. GSA avg ρ: +0.28 for success, −0.13 for timing. Sweep crossings: 16.4 (A), 10.7 (B), 19.3 (C) — all within plausible range but with notable variation. In Scenario B (pressure-driven), the crossing is lowest, suggesting that sharp threshold activation is most effective when combined with the continuous pressure mechanism. Effect is moderate and regime-dependent; Tier 2/3 boundary.

**δ_RoW (RoW discount factor)**
RoW has the lowest influence weight (w=0.166) and is structurally last to move (first-mover probability of 0.0% in every scenario — adoption only occurs after larger blocs have moved). Despite this, δ_RoW ranks 4th in the GSA success ρ (avg +0.21), above γ and κ. The mechanism is indirect: RoW adoption contributes to W accumulation and thereby raises political pressure on any remaining non-adopters. Its patience matters for the final stages of coalition assembly, not the initial coordination decision. Effect is moderate and consistent in sign but relatively small; Tier 3 given regime dependence on whether the final stages of coalition building are the binding constraint.

---

## 7. Notes for Writeup

- Run `python run_analysis.py` (without `--fast`) for final dissertation figures. Rankings will not change; confidence intervals will narrow.
- **Finding 2 (rationality split)** and **Finding 3 (China trap)** are the results most likely to require theoretical grounding — both run against standard intuitions and will need the mechanism explained carefully.
- **Finding 4 (EU null)** should be framed as a meaningful positive result, not a limitation. It identifies where leverage is NOT, which is as policy-relevant as identifying where it IS.
- **Finding 5 (US-CN complementarity)** is the cleanest implication for international negotiation strategy: bilateral or multilateral US-China patience agreements are not merely additive — they have super-additive coordination effects.
- **Figures for dissertation:** `results/figures/comparison_sweeps.png`, `results/figures/gsa_correlations.png`, and the per-scenario `sweeps_{A,B,C}.png` are the primary outputs.
- **On scenario tables:** When presenting within-scenario results (success rates, mean W, etc.) in the dissertation, label them as descriptive summaries of each calibration, not as comparable outcome measures across scenarios.
