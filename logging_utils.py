import json
import logging
import os
import time

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
PREVIEW_CHARS = int(os.getenv("LOG_PREVIEW_CHARS", "300"))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": record.levelname,
        }
        msg = record.getMessage()
        if msg and msg != "":
            payload["message"] = msg
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(LOG_LEVEL)
    _configured = True


def preview(text: str | None) -> str:
    if not text:
        return ""
    return text[:PREVIEW_CHARS]


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    logger.info("", extra={"extra": {"event": event, **fields}})


# Configure on import for simplicity in this challenge project
setup_logging()


