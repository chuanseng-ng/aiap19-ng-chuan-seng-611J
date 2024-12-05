import os
import logging

log_file = "ml_model.log"

# Check if log file exists and delete if exists
if os.path.exists(log_file):
    os.remove(log_file)

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
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
