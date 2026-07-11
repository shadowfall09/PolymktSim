import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    # Disable httpx INFO logs for HTTP requests
    logging.getLogger("httpx").setLevel(logging.WARNING)