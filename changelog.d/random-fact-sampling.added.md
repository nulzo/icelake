`facts.list_for_subject()` and `facts.get_all()` now support opt-in random
sampling with `random=True`. Random samples preserve the existing eligibility
filters and are unpageable; deterministic ID ordering and cursor pagination
remain the default.
