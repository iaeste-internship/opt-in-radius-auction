# ORA Fairness Extension — Design Specification

**Status:** Design proposal prior to implementation  
**Scope:** Passenger-side historical fairness  
**Base mechanism:** Opt-In Radius Auction (ORA)

## 1. Motivation

The current ORA mechanism maximises reported economic welfare subject to passenger capacity, pickup-radius and detour constraints. Driver route cost and the driver’s reported reservation value are included in the economic-welfare objective, and the winner determination problem is solved exactly.

Distributional fairness is currently evaluated only after clearing through Gini and Jain indices. These metrics describe the resulting distribution but do not influence the selected allocation. The purpose of this extension is to introduce passenger fairness directly into the allocation rule while preserving a Groves-compatible mechanism structure.

The extension should satisfy the following principles:

1. Fairness priorities must be public or derived exclusively from observable historical outcomes.
2. They must be fixed before receiving the private reports of the current clearing window.
3. They must not depend on current reported valuations, pickup radii or detour bounds.
4. Economic welfare and the fairness-adjusted allocation objective must remain separately measurable.
5. The winner determination problem must continue to be solved exactly.
6. The extension initially applies only to passengers. Driver-side fairness remains future work.
7. Gini and Jain remain evaluation metrics and are not inserted directly into the allocation objective.

The current measurement design already warns that Gini or Jain alone can assign a favourable fairness score to an allocation in which most passengers remain unserved. For this reason, the allocation bonus is based on individual historical service deprivation rather than directly on a global inequality index.

## 2. Passenger historical state

Each persistent passenger \(i\) has a public historical state:

\[
q_i^t \in [0,1],
\]

where \(t\) denotes the clearing window.

The quantity \(q_i^t\) is interpreted as the passenger’s **discounted service debt** before window \(t\):

- \(q_i^t=0\) means that the passenger has no accumulated service disadvantage.
- A larger \(q_i^t\) means that the passenger has repeatedly requested service without being allocated.
- Values close to one indicate a strong recent history of being unserved.

The initial state is:

\[
q_i^0=0.
\]

Starting new passengers at zero prevents a new account from obtaining an immediate priority advantage.

The state belongs to a persistent passenger identity rather than to an individual request. Therefore, a future simulation must distinguish between:

- a persistent passenger or account identifier; and
- a request identifier for the current window.

The current demand generator creates a new independent population for each run, so a multi-window simulation with persistent identities will be required before historical fairness can be evaluated.

## 3. Public priority

The public priority used by the mechanism is defined directly as:

\[
\pi_i^t=q_i^t.
\]

The notation \(\pi_i^t\) distinguishes the priority used by the allocation rule from the underlying historical state, although they are identical in the initial design.

The priority satisfies:

\[
0\leq \pi_i^t\leq1.
\]

Before the reports of window \(t\) are submitted, the platform publicly commits to:

- the current priority values;
- the state-update rule;
- the memory parameter;
- the fairness weight \(\lambda\).

The priority is not a component of the passenger’s private type:

\[
\theta_i=
\left(
v_i^{solo},
v_i^{share},
r_i^{max},
\delta_i^{max}
\right).
\]

In particular, \(\pi_i^t\) must not depend on any current report of:

- \(v_i^{solo}\);
- \(v_i^{share}\);
- \(r_i^{max}\);
- \(\delta_i^{max}\).

## 4. Historical-state update

Let:

\[
r_i^t=
\begin{cases}
1, & \text{if passenger }i\text{ submits a valid request in window }t,\\
0, & \text{otherwise,}
\end{cases}
\]

and:

\[
x_i^t=
\begin{cases}
1, & \text{if passenger }i\text{ is served in window }t,\\
0, & \text{if passenger }i\text{ requests but is not served.}
\end{cases}
\]

The state is updated according to:

\[
q_i^{t+1}=
\begin{cases}
\rho q_i^t+(1-\rho)(1-x_i^t),
& \text{if }r_i^t=1,\\[4pt]
q_i^t,
& \text{if }r_i^t=0.
\end{cases}
\]

The initial memory parameter is:

\[
\rho=0.8.
\]

Therefore:

### Passenger served

If the passenger requests and is served,

\[
q_i^{t+1}=\rho q_i^t.
\]

Her accumulated service debt decreases but does not necessarily disappear immediately.

### Passenger unserved

If the passenger requests and is not served,

\[
q_i^{t+1}=\rho q_i^t+(1-\rho).
\]

Her priority increases toward one.

### Passenger inactive

If the passenger does not submit a request,

\[
q_i^{t+1}=q_i^t.
\]

Priority is frozen rather than increased or decreased.

The state update is performed only after:

1. the allocation has been selected;
2. all Groves payments have been calculated;
3. the current window has been fully cleared.

The allocation of window \(t\) must use \(q_i^t\), never \(q_i^{t+1}\).

### Operational conventions

The initial implementation adopts the following conventions:

- A request cancelled before clearing does not increase service debt.
- A valid passenger who remains unserved increases her service debt.
- This includes passengers who cannot be served because their reported constraints make all groups involving them infeasible.
- Each passenger state is updated at most once per window.
- Passengers who do not participate in a window retain their previous state.

The last convention is simple but creates a possible intertemporal manipulation concern: a passenger might deliberately submit a request with restrictive constraints to accumulate future priority. This is treated as a limitation of the initial dynamic design.

## 5. Group fairness bonus

For a feasible group \(g\), let \(P(g)\) denote its passenger set.

The group fairness bonus is:

\[
B_t(g)=\sum_{i\in P(g)}\pi_i^t.
\]

The bonus is additive across passengers and groups.

For an allocation \(a\),

\[
F_t(a)
=
\sum_{g\in a}B_t(g).
\]

Because each passenger can appear in at most one selected group,

\[
F_t(a)
=
\sum_i \pi_i^t x_i(a),
\]

where \(x_i(a)=1\) if passenger \(i\) is served in allocation \(a\), and zero otherwise.

The group bonus is not divided by the group size. Serving three high-priority passengers provides three individual fairness credits, rather than the same credit as serving one passenger.

This additive definition preserves the existing weighted set-packing structure: each candidate group continues to have a fixed weight before the WDP is solved.

## 6. Economic welfare

The economic welfare of a feasible group remains unchanged:

\[
W^{econ}(g)
=
\sum_{i\in P(g)}
\widehat v_i\bigl(\operatorname{mode}(g)\bigr)
-
\operatorname{routeCost}(g)
-
\widehat{res}_{j(g)}.
\]

The economic welfare of an allocation is:

\[
W^{econ}(a)
=
\sum_{g\in a}W^{econ}(g).
\]

This definition is already implemented by `group_welfare`: passenger valuations enter positively, while route cost and the selected driver’s reservation value enter negatively. Since a driver can be selected in at most one group, the reservation value is subtracted once whenever that driver participates.

The fairness extension must not modify the meaning of `group_welfare`.

## 7. Augmented allocation objective

For a public fairness weight:

\[
\lambda\geq0,
\]

define the augmented weight of a group:

\[
\widetilde W_t(g)
=
W^{econ}(g)
+
\lambda B_t(g).
\]

The augmented objective of an allocation is:

\[
\Phi_t(a)
=
W^{econ}(a)
+
\lambda F_t(a).
\]

Equivalently,

\[
\Phi_t(a)
=
\sum_{g\in a}
\left[
W^{econ}(g)+\lambda B_t(g)
\right].
\]

The fairness-aware allocation is:

\[
a_t^\lambda
\in
\arg\max_{a\in\mathcal A(\widehat\theta)}
\Phi_t(a).
\]

The WDP remains an exact maximum-weight set-packing problem. The feasibility constraints are unchanged:

- each passenger appears in at most one selected group;
- each driver appears in at most one selected group;
- vehicle capacity is respected;
- passenger pickup-radius constraints are respected;
- passenger detour constraints are respected.

The current WDP already solves the set-packing objective exactly and raises an error when the solver does not return an optimal solution. The fairness extension must preserve this behaviour.

## 8. Separation of reported quantities

Every clearing result must distinguish between:

### Economic welfare

\[
W^{econ}(a_t^\lambda).
\]

This measures the economic value created using passenger valuations, driver route costs and driver reservation values.

### Fairness bonus

\[
F_t(a_t^\lambda).
\]

This measures the total public historical priority served.

### Augmented objective

\[
\Phi_t(a_t^\lambda)
=
W^{econ}(a_t^\lambda)
+
\lambda F_t(a_t^\lambda).
\]

Only \(\Phi_t\) determines the allocation.

Experimental results must not label \(\Phi_t\) as social or economic welfare. The paper should report the economic-welfare loss associated with increasing \(\lambda\), separately from the resulting fairness improvement.

## 9. Groves payments

The allocation and payment rules must use the same augmented objective.

Let:

\[
\kappa_t(a)=\lambda F_t(a)
\]

be the public allocation-dependent fairness term.

For an agent \(k\), define the Clarke payment, with positive values representing a payment to the platform, as:

\[
p_k
=
\max_{a\in\mathcal A_{-k}}
\left[
\sum_{\ell\neq k}
\widehat v_\ell(a)
+
\kappa_t(a)
\right]
-
\left[
\sum_{\ell\neq k}
\widehat v_\ell(a_t^\lambda)
+
\kappa_t(a_t^\lambda)
\right].
\]

For drivers, the same pivot calculation is used with the project’s opposite money-flow convention: positive driver earnings represent a payment from the platform to the driver.

The following quantities must remain identical in every counterfactual solve:

- the priority vector \(\pi^t\);
- the fairness weight \(\lambda\);
- the state-update parameters;
- the definition of the augmented objective.

It would not be valid to allocate by maximising \(\Phi_t\) while computing payments from economic welfare alone.

The payment rule must not truncate negative passenger payments. Under the fairness-aware Groves mechanism, a high-priority passenger may receive a subsidy. Truncating that subsidy at zero would change the Groves payment rule and could invalidate the incentive argument.

## 10. Incentive compatibility

Conditional on the predetermined public state \(\pi^t\), the fairness term is independent of the current private reports.

For fixed reports of all other agents, truthful reporting makes the mechanism maximise:

\[
v_k(a)
+
\sum_{\ell\neq k}\widehat v_\ell(a)
+
\kappa_t(a)
\]

over the relevant allocation set.

The final Clarke counterfactual term is independent of agent \(k\)’s current report. Therefore, the usual Groves argument continues to apply to the valuation and driver-reservation components.

For passenger pickup-radius and detour reports, the existing hard-constraint interpretation remains necessary:

- under-reporting removes truly acceptable allocations;
- over-reporting can add allocations outside the passenger’s true acceptable set;
- an allocation violating a true hard constraint is treated as non-implementable.

The intended proposition is:

> **Proposition.** Conditional on a public priority state fixed before the current reports, exact maximisation of the augmented objective together with the corresponding Groves payments preserves dominant-strategy truthful reporting within each clearing window.

This is a **static, per-window claim**.

The extension does not establish dynamic incentive compatibility across windows. A current report or allocation can influence the passenger’s future priority, so a forward-looking passenger may strategically trade current utility for future priority.

The paper must therefore state that the result applies:

- conditional on the predetermined state;
- within each clearing window;
- under myopic per-window preferences.

## 11. Individual rationality

With exact augmented-objective maximisation and the corresponding Groves payments, the allocation available when an agent is absent remains feasible when the agent is present but left unallocated.

Therefore, under truthful reporting:

\[
\Phi_t(a_t^\lambda)
\geq
\max_{a\in\mathcal A_{-k}}\Phi_t(a).
\]

The agent’s Groves utility is consequently non-negative.

The implementation must nevertheless re-test:

- passenger individual rationality;
- driver individual rationality;
- minimum allocated-driver utility;
- driver IR violation rate.

The current mechanism reports zero driver IR violations over the five preliminary seeds, with positive mean and minimum allocated-driver utility.

## 12. Efficiency and budget implications

For:

\[
\lambda=0,
\]

the mechanism must reproduce the current ORA allocation and payments exactly.

For:

\[
\lambda>0,
\]

the mechanism no longer necessarily maximises economic welfare alone. Instead, it maximises the declared policy objective:

\[
W^{econ}(a)+\lambda F_t(a).
\]

The paper should distinguish between:

- exact economic efficiency at \(\lambda=0\);
- exact optimisation of the augmented objective at \(\lambda>0\);
- the economic price of fairness.

The price of fairness can be reported as:

\[
\operatorname{PoF}(\lambda)
=
\frac{
W^{econ}(a^0)-W^{econ}(a^\lambda)
}{
W^{econ}(a^0)
}.
\]

Fairness-aware payments may increase the platform deficit because:

- passenger payments may fall;
- passenger subsidies may appear;
- driver payments may change.

Budget balance remains an empirical outcome rather than a guaranteed property.

## 13. Initial parameter values

The first fairness sweep will use:

\[
\lambda\in\{0,\ 1,\ 2.5,\ 5,\ 10\}.
\]

Interpretation:

- \(\lambda=0\): current ORA control.
- \(\lambda=1\): weak fairness intervention.
- \(\lambda=2.5\): low-to-moderate intervention.
- \(\lambda=5\): intervention comparable to the current mean driver reservation value.
- \(\lambda=10\): strong fairness intervention.

A larger value such as:

\[
\lambda=20
\]

may later be used as a stress test, but not as a primary specification.

The initial state-memory parameter is:

\[
\rho=0.8.
\]

The first experiment will vary \(\lambda\) while holding \(\rho\) fixed. A later robustness check may compare different values of \(\rho\).

## 14. Experimental design

Historical fairness requires a sequence of clearing windows rather than independent single-window markets.

The initial experiment will use:

- 50 windows per seed;
- 10 burn-in windows;
- 40 measured windows;
- the same market realisations for all \(\lambda\) values;
- independent priority state for each experimental arm;
- persistent passenger identities;
- state updates after each clearing;
- at least 10 seeds for the main experiment.

The current `run_arms.py` executes one independently generated market per seed, so it must later be extended or replaced by a sequential simulation runner.

## 15. Metrics

The fairness experiment should report at least:

### Economic outcomes

- economic welfare;
- price of fairness relative to \(\lambda=0\);
- match rate;
- pooling rate;
- mean wait over served passengers;
- mean wait over all requesting passengers;
- mean detour;
- platform deficit.

### Historical-fairness outcomes

- total fairness bonus;
- mean priority of served passengers;
- mean priority of unserved passengers;
- service rate by priority quartile;
- maximum passenger priority;
- fraction of requests made by passengers above a high-priority threshold;
- fraction of high-priority passengers served;
- average number of consecutive unsuccessful requests.

### Existing distributional metrics

- Gini waiting time;
- Jain waiting time;
- Gini driver earnings;
- Jain driver earnings.

Gini and Jain remain descriptive evaluation metrics. The mechanism does not guarantee that either index improves for every seed or every value of \(\lambda\).

## 16. Required tests

Before experimental use, the implementation must pass the following tests.

### Historical-state tests

1. Every new passenger starts with \(q_i^0=0\).
2. An unserved request increases \(q_i\).
3. A served request decreases \(q_i\).
4. An inactive passenger keeps the same \(q_i\).
5. The state always remains in \([0,1]\).
6. The state is updated once and only after clearing.

### Bonus tests

7. A group bonus equals the sum of its passenger priorities.
8. An allocation bonus equals the sum of the selected-group bonuses.
9. A passenger appearing in one group contributes her priority exactly once.
10. The economic-welfare function remains unchanged.

### Objective tests

11. The augmented objective satisfies:

\[
\Phi_t(a)=W^{econ}(a)+\lambda F_t(a).
\]

12. With \(\lambda=0\), ORA allocations, payments and welfare reproduce the current implementation.
13. Fairness never makes an infeasible group admissible.
14. Capacity, radius and detour constraints remain unchanged.

### Allocation-threshold test

Construct two incompatible passenger alternatives \(A\) and \(B\) such that:

\[
W_A^{econ}>W_B^{econ}
\]

but:

\[
\pi_B>\pi_A.
\]

The winner should change exactly when:

\[
\lambda
>
\frac{
W_A^{econ}-W_B^{econ}
}{
\pi_B-\pi_A
}.
\]

### Payment tests

15. Every main and exclusion solve uses the same priorities and \(\lambda\).
16. A small hand-built market reproduces manually calculated augmented Groves payments.
17. Negative passenger payments are preserved rather than truncated.
18. Passenger IR is re-tested.
19. Driver IR is re-tested.
20. The current mechanism is recovered exactly at \(\lambda=0\).

### Sequential tests

21. Window \(t\) uses \(q^t\), not \(q^{t+1}\).
22. Priority states do not leak between seeds.
23. Priority states do not leak between mechanism arms.
24. A repeatedly unserved passenger eventually receives greater allocation priority.
25. Experimental results are reproducible under a fixed random seed.

## 17. Expected code changes

No code is modified until this design is approved.

The expected implementation changes are:

### New file

`src/ora/fairness.py`

Responsibilities:

- historical state representation;
- state update;
- passenger priority lookup;
- group fairness bonus;
- allocation fairness bonus;
- augmented group weight.

### Existing files

`src/ora/wdp.py`

- accept a priority mapping and \(\lambda\);
- maximise augmented group weights;
- preserve exact optimisation;
- return the augmented objective separately from economic welfare.

`src/ora/payments.py`

- pass the same priorities and \(\lambda\) to all solves;
- use the augmented objective in every Groves counterfactual;
- expose economic welfare, fairness bonus and augmented objective separately.

`src/ora/metrics.py`

- add fairness-specific metrics;
- retain existing Gini and Jain calculations;
- retain driver-utility and IR metrics.

`scripts/run_arms.py` or a new sequential runner

- execute multiple clearing windows;
- maintain persistent passenger state;
- sweep over \(\lambda\);
- use identical demand realisations across treatments.

`src/ora/demand.py` or a new simulation module

- support persistent passenger identities;
- distinguish accounts from individual requests.

### Files not expected to change initially

- `routing.py`;
- `feasibility.py`;
- `grouping.py`;
- the definition of `group_welfare`.

## 18. Frozen design decisions

The initial version adopts the following fixed decisions:

1. Fairness applies only to passengers.
2. Priority represents discounted historical non-service.
3. Priority is public and fixed before current reports.
4. New passengers start with zero priority.
5. Inactive passengers retain their state.
6. The group bonus is the sum of passenger priorities.
7. Gini and Jain are not inserted into the WDP.
8. Economic welfare remains separately reported.
9. All Groves counterfactuals use the augmented objective.
10. Dynamic strategy-proofness is not claimed.
11. The initial values are \(\rho=0.8\) and \(\lambda\in\{0,1,2.5,5,10\}\).
12. Driver fairness is reserved for a later extension.

## 19. Main limitation

The main theoretical limitation is intertemporal manipulation.

Although the priority entering the current window is public and independent of current reports, future priority depends on whether the passenger is served. A forward-looking passenger may attempt to influence future priority through current participation, cancellation or restrictive feasibility reports.

The first implementation therefore studies a sequence of static Groves mechanisms with myopic agents. It does not claim to implement a dynamically incentive-compatible mechanism.
