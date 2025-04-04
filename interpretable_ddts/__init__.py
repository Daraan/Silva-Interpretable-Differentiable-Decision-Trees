from ray_utilities.nice_logging import nicer_logging

from .ddt_setup import DDTSetup

logger = nicer_logging(__name__, "INFO")


__all__ = ["DDTSetup"]
