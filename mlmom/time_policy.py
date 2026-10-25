"""Finite accounting sentinel; hard deadlines are enforced only when opted in."""
UNBOUNDED_SECONDS = 1e12


def timeouts_enabled(config):
    return getattr(config, 'timeouts_enabled', True)


def agent_payload(value, *, timed):
    """JSON-facing budgets must describe absence of a deadline explicitly."""
    if isinstance(value, dict):
        return {key: None if not timed and key in {
                    'remaining_seconds', 'research_seconds', 'remaining_cycle_seconds', 'deadline'}
                else agent_payload(item, timed=timed) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [agent_payload(item, timed=timed) for item in value]
    return value
