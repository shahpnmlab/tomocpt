
from dataclasses import dataclass, fields, MISSING
from enum import Enum
from typing import Optional

@dataclass
class PrepdataConfig:
    DESIRED_PARTICLE_PIXELS: int = 10

