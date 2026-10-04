# Implementing Hamiltonian-Friendly Orbit Schemes in Agama

This note is a kickoff guide for future implementation work in Agama source code.
Primary goals:

1. Add a Cartesian leapfrog method as a new orbit integrator option.
2. Add another method aimed at preserving the Jacobi constant in the rotating frame.

This is scoped as a practical engineering roadmap, not a theory review.

## Why this note exists

Agama currently offers high-order non-symplectic steppers for orbit integration, but not a dedicated symplectic leapfrog method. For long integrations in near-Hamiltonian settings, leapfrog often gives better long-term bounded energy behavior than adaptive RK methods at similar cost per force evaluation.

For rotating-frame dynamics, energy is not conserved in the inertial-frame sense; the key invariant is the Jacobi integral. A second method should be designed around that invariant.

## Scope for phase 1 and phase 2

## Phase 1: Cartesian leapfrog (recommended first target)

- Coordinate system: Cartesian only.
- Potential: time-independent preferred at first.
- Rotating frame: initially allow Omega = 0 only, unless rotating support is explicitly designed.
- Timestep: fixed step preferred for symplectic behavior.

## Phase 2: Rotating-frame method targeting Jacobi preservation

- Coordinate system: start in Cartesian rotating formulation.
- Invariant target: Jacobi constant.
- Timestep: fixed step initially.

## Important expectation setting

- Leapfrog does not exactly conserve Hamiltonian at finite timestep.
- With fixed timestep and suitable splitting, it usually produces bounded oscillatory error rather than secular drift.
- Adaptive timestep generally weakens strict symplectic properties.

## Relevant Agama architecture points

These are the integration seams identified in the earlier inspection:

- New steppers are classes derived from BaseOdeStepper in math_ode.h.
- Orbit integration method selection is done via OrbitIntParams::Method enum and switch in orbit.h.
- Runtime functions consume dense output through getSol(timeOffset), so each stepper must provide valid interpolation over each completed step.
- Second-order eval2 is only implemented for Cartesian orbit integrator specialization; non-Cartesian eval2 is intentionally unsupported.

Implication: adding a stepper is not only about doStep(), but also interpolation and runtime compatibility.

## Proposed implementation design

## A. New Cartesian Leapfrog stepper

Add class in math_ode.h/cpp, for example:

- OdeStepperLeapfrogKDK

Suggested constructor signature:

- OdeStepperLeapfrogKDK(const IOdeSystem2ndOrder& odeSystem, double timeStep)

Key fields:

- const IOdeSystem2ndOrder& odeSystem
- int NDIM
- double fixedTimeStep
- double prevTimeStep
- std::vector<double> state

State layout suggestion:

- current x and v at end of last step
- cached x0 and v0 at beginning of last step for interpolation
- cached a0 and a1 if needed

KDK step pattern:

1. Kick half step: v_{n+1/2} = v_n + 0.5 dt a(x_n)
2. Drift full step: x_{n+1} = x_n + dt v_{n+1/2}
3. Kick half step: v_{n+1} = v_{n+1/2} + 0.5 dt a(x_{n+1})

Interpolation strategy for getSol:

- Keep it simple and robust at first.
- Position interpolation: quadratic using x_n, v_{n+1/2}, and endpoint acceleration approximation.
- Velocity interpolation: linear between v_n and v_{n+1} or midpoint-based piecewise rule.
- Minimum acceptable: endpoint-consistent interpolation that does not violate runtime sampling expectations.

Do not over-engineer interpolation in v1. Reliability is more important than high polynomial order.

## B. Orbit method plumbing

In orbit.h:

- Add method enum value, for example LEAPFROG.
- Extend OrbitIntParams to carry fixed timestep for leapfrog.

Suggested parameter extension:

- double fixedTimeStep;
- default could be 0 meaning auto, but for symplectic use, require explicit nonzero dt.

In BaseOrbitIntegrator constructor switch:

- Instantiate OdeStepperLeapfrogKDK when method == LEAPFROG.

Guardrails:

- If method is leapfrog and chosen coordinate type is non-Cartesian, throw clear runtime error in init or constructor path.
- If rotating frame support is not implemented yet, require Omega == 0 with a clear error.

## C. Jacobi-preserving rotating-frame method

This should be a separate method option, not hidden behind leapfrog flags.

Candidate approaches:

1. Split-Hamiltonian integrator in rotating canonical variables
- Most principled route.
- Requires careful formulation of Coriolis and centrifugal terms in canonical form.

2. Implicit midpoint in rotating frame
- Not symplectic for arbitrary splitting choices, but time-reversible and often very good at preserving invariants in practice.
- May preserve a modified integral very well.

3. Discrete-gradient or projection method
- Explicitly correct step to preserve Jacobi constraint.
- Adds solve/projection overhead and implementation complexity.

Recommendation for v1:

- Implement a dedicated rotating-frame KDK-like split only after writing down exact canonical equations and validating the map is symplectic/time-reversible.
- If schedule is tight, start with implicit midpoint as a pragmatic Jacobi-stable baseline method and evaluate.

## Validation plan

## Test set 1: Static potential, Omega = 0, Cartesian

- Compare DOP853 vs Leapfrog on long integrations.
- Track relative energy error:
	- max absolute deviation
	- drift slope via linear fit over time
- Expect leapfrog to show bounded oscillatory envelope at fixed dt.

## Test set 2: Rotating frame, Omega != 0

- Track Jacobi constant error for current methods and candidate new method.
- Evaluate:
	- max |Delta C_J|
	- drift slope
	- dependence on dt

## Test set 3: Time reversibility sanity check

- Integrate forward T, negate dt, integrate backward T.
- Measure phase-space mismatch norm.
- Leapfrog and symmetric methods should perform strongly here.

## Test set 4: Runtime sampling consistency

- Use trajectory sampling with samplingInterval > 0 and 0.
- Ensure no anomalies from getSol interpolation near step boundaries.

## Suggested implementation order

1. Add enum/plumbing and skeleton class for leapfrog.
2. Implement fixed-step doStep for Cartesian eval2 path.
3. Implement simple consistent getSol interpolation.
4. Add runtime guardrails for unsupported combinations.
5. Build and run basic trajectory checks.
6. Add diagnostics script to compare energy behavior versus DOP853.
7. Only then design rotating Jacobi-oriented method.

## Risks and pitfalls

- Trying to keep adaptive timestep while expecting strong symplectic behavior.
- Underestimating interpolation requirements for runtime functions.
- Extending leapfrog immediately to Cyl/Sph where equations include velocity-coupled terms not suited to naive split.
- Mixing inertial and rotating-frame variable conventions without an explicit canonical derivation.

## Decisions to lock before coding session

1. Leapfrog v1 fixed timestep value selection rule.
2. Whether leapfrog v1 supports Omega != 0 or explicitly forbids it.
3. Exact interpolation formula quality target for getSol.
4. Which rotating-frame method to pursue first: canonical split or implicit midpoint baseline.

## Prep checklist for our coding session

- Have a local Agama build ready and reproducible.
- Prepare one static potential test case and one rotating bar test case.
- Define acceptance thresholds for energy and Jacobi error metrics.
- Decide a default dt grid for convergence checks.

## Deliverables we should target

1. New method option for Cartesian leapfrog in Agama.
2. A small benchmark note comparing long-term invariants vs existing methods.
3. A second method prototype for rotating-frame Jacobi behavior, with explicit limitations documented.

## Notes for future me

- Keep phase 1 small and complete.
- Prioritize correctness and diagnostics over fancy interpolation.
- Avoid broad support claims until tests verify expected invariant behavior.

