from typing import Annotated

import typer

from tomocpt.defaultConfigs.prepdata_config import PrepareDataType
from typer import Option
import pandas as pd
from omegaconf import DictConfig

from tomocpt.labels.star import Star
from tomocpt.labels.mod import Mod
from tomocpt.labels.helpers import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def create_tracking_dataframe(vol_coord_pairs, tomo_path: Path, output_dir: Path,
                              particle_diameter_angst: float) -> pd.DataFrame:
    """
    Create a DataFrame tracking tomograms, labels, and particle information.

    Args:
        vol_coord_pairs: Dictionary mapping tomogram names to particle coordinates
        tomo_path: Path to tomogram directory
        output_dir: Path to output directory
        particle_diameter_angst: Particle diameter in Angstroms

    Returns:
        pd.DataFrame: DataFrame containing tracking information
    """
    tracking_data = []

    for tomo_name, coordinates in vol_coord_pairs.items():
        tomo_file = next(tomo_path.glob(f"*{tomo_name}*"))
        label_file = output_dir / f"{tomo_name}_labels.mrc"

        tracking_data.append({
            'tomogram_path': str(tomo_file),
            'label_path': str(label_file),
            'particle_diameter_angst': particle_diameter_angst,
            'num_particles': len(coordinates)
        })

    df = pd.DataFrame(tracking_data)
    return df


def prepare_picking_imod(input_file: Path ,
                         tomo_path: Path = Option(..., "--tomo-path", "-t", help="Path to tomograms"),
                         output_dir: Path = Option(..., "--output", "-o", help="Output directory"),
                         particle_diameter_angst: float = Option(..., "--length", "-l",
                                                                 help="Particle longest axis length (Å)")
                         ) -> None:
    particle_radius_angst = particle_diameter_angst / 2.0
    tomo_path = Path(tomo_path)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Mod class
    mod = Mod(input_file)

    vol_coord_pairs = mod.get_coordinates()
    matched_data = match_data_to_tomograms(vol_coord_pairs, tomo_path)

    # Create tracking DataFrame
    tracking_df = create_tracking_dataframe(matched_data, tomo_path, output_dir, particle_diameter_angst)

    # Save tracking information
    tracking_csv = output_dir / "imod_picking_tracking.csv"
    tracking_df.to_csv(tracking_csv, index=False)
    logging.info(f"Saved tracking information to {tracking_csv}")

    # Create particle masks
    create_particle_masks(matched_data,
                          radius_ang=particle_radius_angst,
                          tomogram_path=tomo_path,
                          output_path=output_dir,
                          class_id="all")


def prepare_picking_star(input_file: Path,
                         tomo_path: Path,
                         output_dir: Path,
                         class_id: str,
                         particle_diameter_angst: float
                         ) -> None:

    particle_radius_angst = particle_diameter_angst / 2.0
    tomo_path = Path(tomo_path)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    star = Star(input_file)

    # Get particle subset for specific classes
    subset_data, class_id = star.get_particle_subset(class_id)
    matched_data = match_data_to_tomograms(subset_data, tomo_path)

    # Get shifted and scaled coordinates
    vol_coord_pairs = star.get_shifted_scaled_coordinates(matched_data, tomogram_pixel_size=star.pixel_size)

    # Create tracking DataFrame
    tracking_df = create_tracking_dataframe(vol_coord_pairs, tomo_path, output_dir, particle_diameter_angst)

    # Save tracking information
    tracking_csv = output_dir / f"star_picking_tracking_{class_id}.csv"
    tracking_df.to_csv(tracking_csv, index=False)
    logging.info(f"Saved tracking information to {tracking_csv}")

    # Create particle masks
    create_particle_masks(
        vol_coord_pairs,
        radius_ang=particle_radius_angst,
        tomogram_path=tomo_path,
        output_path=output_dir,
        class_id=class_id
    )


def prepare_labels(config:DictConfig=None):
    if config.coordinate_file_type == PrepareDataType.star:
        prepare_picking_star(input_file=config.input_file,
                             tomo_path=config.raw_data_dir,
                             output_dir=config.prepared_data_dir,
                             class_id=config.class_id,
                             particle_diameter_angst=config.particle_length_ang)

    if config.coordinate_file_type == PrepareDataType.imod:
        prepare_picking_imod(input_file=config.input_file,
                             tomo_path=config.raw_data_dir,
                             output_dir=config.prepared_data_dir,
                             particle_diameter_angst=config.particle_length_ang)


if __name__ == '__main__':
    prepare_labels()