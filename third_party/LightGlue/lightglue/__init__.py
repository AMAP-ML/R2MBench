try:
    from .aliked import ALIKED  # noqa
except ImportError:
    pass
try:
    from .disk import DISK  # noqa
except ImportError:
    pass
try:
    from .dog_hardnet import DoGHardNet  # noqa
except ImportError:
    pass
from .lightglue import LightGlue  # noqa
try:
    from .sift import SIFT  # noqa
except ImportError:
    pass
from .superpoint import SuperPoint  # noqa
from .utils import match_pair  # noqa
