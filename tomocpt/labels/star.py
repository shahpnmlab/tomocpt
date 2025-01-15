from pathlib import Path
from typing import Union, Dict, Tuple

import numpy as np
import pandas as pd
import starfile

from tomocpt.logger import get_logger

logging = get_logger()

class Star:
    def __init__(self, star_file_path: Union[str, Path]):
        """
        Initialize Star object to handle RELION STAR file processing.

        Args:
            star_file_path (Union[str, Path]): Path to the STAR file
        """
        self.star_file_path = Path(star_file_path)
        self.star_data = starfile.read(self.star_file_path)
        self._initialize_star_properties()

    def _initialize_star_properties(self):
        """Initialize basic properties from the STAR file."""
        if 'optics' in self.star_data:
            self.particle_data = self.star_data['particles']
            self.pixel_size = self.star_data['optics']['rlnImagePixelSize']
            if 'rlnMicrographName' in self.particle_data:
                self.version = 'rln31'
                self.tomo_label = 'rlnMicrographName'
            elif 'rlnTomoName' in self.particle_data:
                self.version = 'rln50'
                self.tomo_label = 'rlnTomoName'
        else:
            self.version = 'rln30'
            self.tomo_label = 'rlnMicrographName'
            self.particle_data = self.star_data
            self.pixel_size = self.star_data['rlnPixelSize']

    def get_particle_subset(self, class_id: str) -> Tuple[pd.DataFrame, str]:
        """
        Get a subset of particle data based on class ID.

        Args:
            class_id (str): Class identifier, can be 'all' or specific classes like '1:2:3'

        Returns:
            Tuple[pd.DataFrame, str]: Subset of particle data and processed class_id
        """
        if 'all' in class_id:
            subset_data = self.particle_data
            processed_class_id = ''.join(class_id)
        else:
            class_id_list = [int(id) for id in class_id.split(':')]

            for id in class_id_list:
                if id not in self.particle_data['rlnClassNumber'].values:
                    raise ValueError(f"Invalid class ID {id}.")

            subset_data = self.particle_data[self.particle_data['rlnClassNumber'].isin(class_id_list)]

            if len(class_id_list) == 1:
                processed_class_id = str(class_id_list[0])
            else:
                processed_class_id = '_'.join(map(str, class_id_list))

        return subset_data, processed_class_id

    def get_shifted_scaled_coordinates(self, particle_data: pd.DataFrame,
                                       tomogram_pixel_size: Union[pd.Series, float]) -> Dict[str, np.ndarray]:
        """
        Process coordinates with shifts and scaling.

        Args:
            particle_data (pd.DataFrame): Particle data to process
            tomogram_pixel_size (Union[pd.Series, float]): Pixel size of the tomogram

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping tomogram names to coordinate arrays
        """
        xyz_headings = [f'rlnCoordinate{axis}' for axis in "XYZ"]
        if self.version == 'rln30':
            shift_headings = [f'rlnOrigin{axis}' for axis in "XYZ"]
        else:
            shift_headings = [f'rlnOrigin{axis}Angst' for axis in "XYZ"]

        coords = particle_data[xyz_headings].to_numpy()

        # Check if shift headings exist in the dataframe
        missing_shifts = [h for h in shift_headings if h not in particle_data.columns]
        if missing_shifts:
            logging.warn(f"Missing shift headings in particle data: {missing_shifts}. Setting shifts to 0.")
            shifts = np.zeros_like(coords)
        else:
            shifts = particle_data[shift_headings].to_numpy()
            if self.version in ['rln31', 'rln50']:
                # NOTE: This makes it such that one cannot work with a star file with tomograms of different pixel sizes
                # defined a single star file.
                shifts = shifts / self.pixel_size.values[0]

        shifted_coords = coords - shifts

        # Fix: Get single values from both Series
        star_pixel_size = self.pixel_size.values[0]
        tomo_pixel_size = tomogram_pixel_size.iloc[0] if isinstance(tomogram_pixel_size,
                                                                    pd.Series) else tomogram_pixel_size

        scaling_factor = star_pixel_size / tomo_pixel_size

        if float(scaling_factor) < 1:
            scaled_coords = shifted_coords * scaling_factor
        else:
            scaled_coords = shifted_coords / scaling_factor

        subset_data = particle_data.copy()
        subset_data[xyz_headings] = scaled_coords

        # Cast the groupby result to explicitly ensure str keys
        return {str(name): group[xyz_headings].astype(int).values
                for name, group in subset_data.groupby(self.tomo_label)}
