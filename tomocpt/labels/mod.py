from pathlib import Path
from typing import Union, Dict, List
import numpy as np
import imodmodel
import logging

logger = logging.getLogger(__name__)

class Mod:
    def __init__(self, mod_path: Union[str, Path]):
        """
        Initialize Mod object to handle IMOD model file processing.

        Args:
            mod_path (Union[str, Path]): Path to the directory containing .mod files
        """
        self.imod_model_path = Path(mod_path)
        if not self.imod_model_path.exists():
            raise FileNotFoundError(f"Path {self.imod_model_path} does not exist")

        self.model_files = sorted(list(self.imod_model_path.glob("*.mod")))
        if not self.model_files:
            raise ValueError(f"No .mod files found in {self.imod_model_path}")

        logger.info(f"Found {len(self.model_files)} .mod files in {self.imod_model_path}")

    def get_coordinates(self) -> Dict[str, np.ndarray]:
        """
        Read all mod files and return coordinates for each volume.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping volume names to coordinate arrays
        """
        vol_coord_pairs = {}

        for mod_file in self.model_files:
            try:
                # Read the mod file
                df = imodmodel.read(mod_file)

                # Extract ZYX coordinates
                #TODO: Support multiple object ids if preset in the df
                headings = [f"{axis}" for axis in "zyx"]
                if not all(heading in df.columns for heading in headings):
                    logger.warning(f"Missing coordinate columns in {mod_file}")
                    continue

                coords = df[headings].to_numpy()

                # Use the stem of the filename (without extension) as the key
                vol_name = str(mod_file.stem)
                vol_coord_pairs[vol_name] = coords.astype(int)

                logger.info(f"Processed {mod_file.name}: found {len(coords)} coordinates")

            except Exception as e:
                logger.error(f"Error processing {mod_file}: {str(e)}")
                continue

        if not vol_coord_pairs:
            raise ValueError("No valid coordinates found in any mod files")

        return vol_coord_pairs
