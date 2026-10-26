"""Application JSON request logs include the matching response request ID."""
import os

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("PILAR_LOG_LEVEL", "info").lower()
access_log_format = '%(t)s %(m)s %(U)s status=%(s)s duration=%(L)s request_id=%({x-request-id}o)s'


def post_fork(server, worker):
    # Also configure the logger on HUP when the master predates this config file.
    server.log.cfg = server.cfg
    server.log.setup(server.cfg)
    from pilar_logging import configure_logging
    configure_logging()
