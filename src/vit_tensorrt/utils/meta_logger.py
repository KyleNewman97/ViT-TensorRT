import logging
from datetime import datetime
from uuid import uuid4


class MetaLogger:
    def __init__(self):
        self.logger = logging.getLogger(f"{uuid4()}")
        self.logger.setLevel(level=logging.INFO)

        # Add a formatter to prepend the class name to the message
        formatter = CustomFormatter(
            fmt=(
                f"{datetime.now()} %(levelname)s: [ViT-TensorRT]"
                f"[{self.__class__.__name__}] %(message)s"
            )
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


class CustomFormatter(logging.Formatter):
    def __init__(self, fmt: str):
        grey = "\x1b[38;20m"
        green = "\x1b[32;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        self.formatters = {
            logging.DEBUG: logging.Formatter(grey + fmt + reset),
            logging.INFO: logging.Formatter(green + fmt + reset),
            logging.WARNING: logging.Formatter(yellow + fmt + reset),
            logging.ERROR: logging.Formatter(red + fmt + reset),
            logging.CRITICAL: logging.Formatter(bold_red + fmt + reset),
        }

    def format(self, record: logging.LogRecord) -> str:
        return self.formatters[record.levelno].format(record)
