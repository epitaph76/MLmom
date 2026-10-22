"""Finite accounting sentinel; hard deadlines are enforced only when opted in."""
UNBOUNDED_SECONDS = 1e12


def timeouts_enabled(config):
    return getattr(config, 'timeouts_enabled', True)
