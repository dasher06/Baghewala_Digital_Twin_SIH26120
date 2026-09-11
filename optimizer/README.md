# Model 5 — Joint CSS + SRP Optimizer

`joint_optimizer.py` is the prototype decision layer. It does not train an ML model. It evaluates candidate CSS + SRP settings by calling the existing Models 1–4 inference functions and selects the best feasible scenario using a constrained multi-objective score.
