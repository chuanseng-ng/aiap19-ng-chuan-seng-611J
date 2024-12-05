import logging


# Create logger
logger = logging.getLogger(__name__)

# Configure logger
logger.setLevel(logging.INFO)

## Create console handler for output to console
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)

# Create formatter and attach to handler
formatter = logging.Formatter("%(levelname)s - %(message)s")
# ch.setFormatter(formatter)

## Add handler to logger
# logger.addHandler(ch)

# Add file logging
fh = logging.FileHandler("ml_model.log")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
