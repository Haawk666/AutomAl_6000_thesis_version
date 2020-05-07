"""Module to handle forwards compatibility between versions.

.. note::

    Only forwards compatibility is maintained. Opening a project file that was saved in a more recent version is
    generally not guarantied to be possible, but opening old project files in newer version should always be possible.

"""
# Internal imports

# External imports
import logging
# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert(obj, old_version):
    """Return project instance in a compatible state.

    :param obj: The core.SuchSoftware instance to be upgraded.
    :param old_version: The version that 'obj' was saved with.

    :type obj: core.SuchSoftware() instance
    :type old_version: list(<int>)

    :returns project instance in a compatible state.
    :rtype core.SuchSoftware() instance

    """

    # Return obj in a compatible state
    # Return None if not possible!

    if old_version == [0, 1, 0]:
        fresh_obj = None

    else:
        fresh_obj = None

    return fresh_obj

