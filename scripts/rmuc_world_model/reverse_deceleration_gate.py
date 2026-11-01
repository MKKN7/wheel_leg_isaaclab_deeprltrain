"""Slow settling is allowed; backward deceleration must not rebound past target."""
def review(event, complete, steady, overshoot_tolerance_mps=.01):
    old, target = event['old'], event['target']
    if not old < target <= 0:
        return dict(applicable=False)
    if not 0 <= overshoot_tolerance_mps <= .02:
        raise ValueError('Only a small explicit speed-noise tolerance is supported')
    overshoot = max(0., event['overshoot_fraction']) * abs(target-old)
    return dict(applicable=True, passed=bool(complete and steady and
                overshoot <= overshoot_tolerance_mps),
                overshoot_mps=overshoot, tolerance_mps=overshoot_tolerance_mps,
                arrival_deadline_removed=True)
