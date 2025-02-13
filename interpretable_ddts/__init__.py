import logging

from ray_utilities import utilities_handler

from .ddt_setup import DDTSetup

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    logger.addHandler(utilities_handler)

__all__ = ["DDTSetup"]
