__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "08/10/2021"

import os as _os

project = _os.path.basename(_os.path.dirname(_os.path.abspath(__file__)))

try:
    from ._version import __date__ as date  # noqa
    from ._version import (
        version,
        version_info,
        hexversion,
        strictversion,
        dated_version,
    )  # noqa
except ImportError:
    raise RuntimeError(
        "Do NOT use %s from its sources: build it and use the built version"
        % project
    )
from ._multianalyzer import MultiAnalyzer
from .file_io import ID22_bliss_parser, topas_parser
