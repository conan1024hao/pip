from .blip2 import Blip2_PIP
from .InstructBlip import InstructBlip_PIP
try:
    from .qwen25vl import Qwen25VL_PIP
except ImportError:
    Qwen25VL_PIP = None
