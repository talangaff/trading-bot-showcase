"""Entry point for the hourly 'This Week' refresh (run by the systemd timer).

    python -m webapp.refresh_week            # refresh all days in the window
    python -m webapp.refresh_week 2026-08-27 # refresh a single day
"""

import sys

from . import week


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    try:
        if argv and argv[0] == "near":
            status = week.refresh_near()
        elif argv:
            status = week.refresh_day(argv[0])
        else:
            status = week.refresh_all(verbose=True)
    except week.RefreshBusy:
        print("another refresh is already running; skipping", flush=True)
        return 0
    if status.get("last_error"):
        print("completed with errors: {}".format(status["last_error"]), flush=True)
    else:
        print("refresh complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
