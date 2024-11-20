
from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional
from omegaconf import MISSING

@dataclass
class PrepdataConfig:
    DESIRED_PARTICLE_PIXELS: int = 10

