from .stats import load_hcp_stats


def run_hcp_eval(*args, **kwargs):
    from .runner import run_hcp_eval as _run_hcp_eval

    return _run_hcp_eval(*args, **kwargs)


__all__ = ["run_hcp_eval", "load_hcp_stats"]
