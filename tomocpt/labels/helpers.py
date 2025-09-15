from pathlib import Path
from typing import Union, Dict
from itertools import chain

from tomocpt.constants import RAW_LABEL_FNAME_TEMPLATE
from tqdm import tqdm
import functools
import mrcfile
import numpy as np
import pandas as pd


from tomocpt.labels.mod import Mod
from tomocpt.labels.star import Star
from tomocpt.logger import get_logger

logging = get_logger()

def get_tomogram_pixel_size(tomogram_path: Path) -> float:
    """
    Read pixel size from the first tomogram in the directory.

    Args:
        tomogram_path: Path to tomogram directory

    Returns:
        float: Pixel size in Angstroms

    Note:
        This function assumes all tomograms in the directory have
        the same pixel size, which may not be true for all datasets.
    """
    tomo_files = list(Path(tomogram_path).glob('*.mrc'))
    if not tomo_files:
        raise FileNotFoundError(f"No .mrc files found in {tomogram_path}")

    with mrcfile.open(tomo_files[0]) as mrc:
        tomogram_pixel_size = float(mrc.voxel_size.x)

    return tomogram_pixel_size

def match_data_to_tomograms(particle_data: Union[pd.DataFrame, dict],
                            tomogram_path: Union[str, Path]) -> Union[pd.DataFrame, dict]:
    """
    Match particle data to tomograms in the specified folder.

    Args:
        particle_data: Either a DataFrame with tomogram names or a dict with tomogram data
        tomogram_path: Path to tomogram folder
        tomo_label: Column name containing tomogram names (for DataFrame input)

    Returns:
        Either a matched DataFrame or dict depending on input type
    """
    COMMON_TOMO_COLUMNS = ['rlnMicrographName', 'rlnTomoName']
    folder_with_tomograms = Path(tomogram_path)
    tomograms_in_folder = list(folder_with_tomograms.glob('*.mrc'))
    tomogram_names_in_folder = sorted([tomogram.stem for tomogram in tomograms_in_folder])

    # Create a mapping dictionary to store the matched names
    name_mapping = {}

    def match_tomogram_name(x):
        x = Path(x).stem

        if x in name_mapping:
            return name_mapping[x]

        if x in tomogram_names_in_folder:
            name_mapping[x] = x
            return x

        possible_matches = [name for name in tomogram_names_in_folder
                            if name.startswith(x) and (name[len(x):len(x) + 1] in ['_', '-', ''] or name == x)]

        if len(possible_matches) == 1:
            logging.info(f"Close match found for tomogram {x}: {possible_matches[0]}")
            name_mapping[x] = possible_matches[0]
            return possible_matches[0]
        elif len(possible_matches) > 1:
            logging.info(f"Multiple possible matches found for tomogram {x}: {possible_matches}")
            name_mapping[x] = None
            return None
        else:
            logging.info(f"No match found for tomogram {x} in the folder {folder_with_tomograms}.")
            name_mapping[x] = None
            return None

    if isinstance(particle_data, pd.DataFrame):
        # Get the 0th element of the COMMON_TOMO_COULMNS list because the comparison in
        # line 67 will break.
        tomo_label = [col for col in COMMON_TOMO_COLUMNS if col in particle_data.columns][0]
        particle_data = particle_data.copy()
        if tomo_label not in particle_data.columns:
            logging.critical(
                f"Column '{tomo_label}' not found in particle_data DataFrame. Available columns are: {list(particle_data.columns)}")
            raise KeyError(f"Critical error: Column '{tomo_label}' not found in DataFrame")

        particle_data['matched_tomogram'] = particle_data[tomo_label].apply(match_tomogram_name)
        matched_data = particle_data[particle_data['matched_tomogram'].notna()].copy()

        if 'matched_tomogram' in matched_data.columns:
            matched_data[tomo_label] = matched_data['matched_tomogram']
            matched_data.drop('matched_tomogram', axis=1, inplace=True)

#        logger.info(f"Found matches for {len(matched_data)} out of {len(particle_data)} particles")
        return matched_data

    elif isinstance(particle_data, dict):
        # Handle dictionary input
        matched_data = {}
        for key, value in particle_data.items():
            matched_name = match_tomogram_name(key)
            if matched_name is not None:
                matched_data[matched_name] = value
#        logger.info(f"Found matches for {len(matched_data)} out of {len(particle_data)} tomograms")
        return matched_data

    else:
        raise TypeError("particle_data must be either a pandas DataFrame or a dictionary")


@functools.cache
def generate_gaussian_sphere(radius: float, grid_size: int,
                             #0.33 and 0.66 for small and large gives the same results as the old code when sigma_radius_equivalance=2 and fraction_of_small_vs_large=0.5
                             fraction_of_radius_small_gaussian: float = 0.33,
                             fraction_of_radius_large_gaussian: float = 0.66,
                             fraction_of_small_vs_large: float = 0.5,
                             sigma_radius_equivalance:float = 2, #2 sigma ~ 1 radius (95% of weight)
                             ) -> np.ndarray:
    #TODO: generalise the gaussian generation form and move default values to config
    """
    Generate a sphere with Gaussian falloff.

    Args:
        radius (float): Radius of the particle in pixels
        grid_size (int): Size of the grid
        fraction_of_radius_small_gaussian (float): which is the fraction of the radius to use as width for the small gaussian
        fraction_of_radius_large_gaussian (float): which is the fraction of the radius to use as width for the large gaussian
        fraction_of_small_vs_large (float): The proportion of small gaussian in the final label. large gaussian proportion is 1-small_prop
        sigma_radius_equivalance (float): How much sigma do we convey is equivalent to the radius. This is an arbitrary decsion, but since 95% of the mass is within mean +/- 2*sigma, 2 makes sense

    Returns:
        np.ndarray: Generated sphere
    """

    x = np.linspace(-radius, radius, grid_size)
    y = np.linspace(-radius, radius, grid_size)
    z = np.linspace(-radius, radius, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)

    distance_from_center = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    sigma_large = fraction_of_radius_large_gaussian * radius / sigma_radius_equivalance
    sigma_small = fraction_of_radius_small_gaussian * radius / sigma_radius_equivalance
    gaussian_small = np.exp(-(distance_from_center ** 2) / (2 * sigma_small ** 2))
    gaussian_large = np.exp(-(distance_from_center ** 2) / (2 * sigma_large ** 2))

    combined_gaussian = fraction_of_small_vs_large * gaussian_small + (1 - fraction_of_small_vs_large) * gaussian_large
    normed_gaussian = combined_gaussian / combined_gaussian.max()
    normed_gaussian[distance_from_center > radius] = 0
    return normed_gaussian

"""
def generate_gaussian_sphere(radius: float, grid_size: int,
                             falloff_ratio: float = 1, accent_ratio: float = 0.5,
                             falloff_value: float = 0.01, accent_value: float = 0.01) -> np.ndarray:
    #TODO: generalise the gaussian generation form and move default values to config

    Generate a sphere with Gaussian falloff.

    Args:
        radius (float): Radius of the sphere
        grid_size (int): Size of the grid
        falloff_ratio (float): Ratio for Gaussian falloff
        accent_ratio (float): Ratio for accent
        falloff_value (float): Falloff value
        accent_value (float): Accent value

    Returns:
        np.ndarray: Generated sphere

    sigma_falloff = (falloff_ratio * radius) / np.sqrt(-2 * np.log(falloff_value))
    sigma_accent = (accent_ratio * radius) / np.sqrt(-2 * np.log(accent_value))

    x = np.linspace(-radius, radius, grid_size)
    y = np.linspace(-radius, radius, grid_size)
    z = np.linspace(-radius, radius, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)

    distance_from_center = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    gaussian_falloff = np.exp(-(distance_from_center ** 2) / (2 * sigma_falloff ** 2))
    gaussian_accent = np.exp(-(distance_from_center ** 2) / (2 * sigma_accent ** 2))

    combined_gaussian = gaussian_falloff + gaussian_accent
    normed_gaussian = combined_gaussian / combined_gaussian.max()
    normed_gaussian[distance_from_center > radius] = 0

    return normed_gaussian
"""
def create_particle_masks(vol_coord_pairs: Dict[str, np.ndarray],
                          radius_ang: float, tomogram_path: Union[str, Path],
                          output_path: Union[str, Path], class_id: str):
    """
    Create particle masks for all coordinates.

    Args:
        vol_coord_pairs (Dict[str, np.ndarray]): Dictionary of coordinates per tomogram
        radius_ang (float): Radius in Angstroms
        tomogram_path (Union[str, Path]): Path to tomogram folder
        output_path (Union[str, Path]): Output path for masks
        class_id (str): Class identifier
    """
    precision = np.float32
    one_tomo_partial = sorted(list(vol_coord_pairs.keys()))[0]
    one_tomo_fname = list(Path(tomogram_path).glob(f"{one_tomo_partial}*.mrc"))[0]

    with mrcfile.open(one_tomo_fname) as mrc:
        tomo_dim = mrc.data.shape
        tomogram_pixel_size = mrc.voxel_size.x

    radius_px = round(radius_ang / tomogram_pixel_size)

    sphere_box_size = int((2 * radius_px) + 1)
    if sphere_box_size % 2 == 0:
        sphere_box_size += 1

    sphere = generate_gaussian_sphere(radius=radius_px, grid_size=sphere_box_size).astype(precision)

    for filename, points in tqdm(vol_coord_pairs.items(), desc="Generating particle masks"):
        segmentation_vol = np.zeros(tomo_dim, dtype=precision)

        for point in points:
            x, y, z = point
            zmin, zmax = int(z - radius_px), int(z + radius_px) + 1
            ymin, ymax = int(y - radius_px), int(y + radius_px) + 1
            xmin, xmax = int(x - radius_px), int(x + radius_px) + 1

            if (zmin <= 0 or zmax >= tomo_dim[0] or
                    ymin <= 0 or ymax >= tomo_dim[1] or
                    xmin <= 0 or xmax >= tomo_dim[2]):
                continue

            segmentation_vol[zmin:zmax, ymin:ymax, xmin:xmax] = np.maximum(
                segmentation_vol[zmin:zmax, ymin:ymax, xmin:xmax], sphere)

        output_dir = Path(output_path) / f'class_{class_id}' / 'labels'
        output_dir.mkdir(parents=True, exist_ok=True)

        mrcfile.write(
            output_dir / f'{filename}.labels.mrc',
            segmentation_vol,
            voxel_size=tomogram_pixel_size,
            overwrite=True
        )

def prepare_picking_imod(input_file: Path ,
                         tomo_path: Path ,
                         output_dir: Path ,
                         particle_diameter_angst: float
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
    tracking_df = create_tracking_dataframe(matched_data, tomo_path, output_dir, particle_diameter_angst, class_id)

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
    tomogram_voxel_size = get_tomogram_pixel_size(tomo_path)
    vol_coord_pairs = star.get_shifted_scaled_coordinates(matched_data, tomogram_pixel_size=tomogram_voxel_size)

    # Create tracking DataFrame
    tracking_df = create_tracking_dataframe(vol_coord_pairs, tomo_path, output_dir, particle_diameter_angst, class_id)

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

def create_tracking_dataframe(vol_coord_pairs, tomo_path: Path, output_dir: Path,
                              particle_diameter_angst: float, class_id:str) -> pd.DataFrame:
    """
    Create a DataFrame tracking tomograms, labels, and particle information.

    Args:
        vol_coord_pairs: Dictionary mapping tomogram names to particle coordinates
        tomo_path: Path to tomogram directory
        output_dir: Path to output directory
        particle_diameter_angst: Particle diameter in Angstroms
        class_id: the class id of the particle
    Returns:
        pd.DataFrame: DataFrame containing tracking information
    """
    tracking_data = []

    for tomo_name, coordinates in vol_coord_pairs.items():
        # Search for both .mrc and .rec files
        tomo_file = next(chain(
            tomo_path.glob(f"*{tomo_name}*.mrc"),
            tomo_path.glob(f"*{tomo_name}*.rec")
        ), None)

        if tomo_file is None:
            raise FileNotFoundError(f"No .mrc or .rec file found for tomogram: {tomo_name}")

        label_file = output_dir / f"class_{class_id}" / "labels" / Path(RAW_LABEL_FNAME_TEMPLATE%str(tomo_name))

        tracking_data.append({
            'tomogram_path': str(tomo_file),
            'label_path': str(label_file),
            'particle_diameter_angst': particle_diameter_angst,
            'num_particles': len(coordinates)
        })

    df = pd.DataFrame(tracking_data)
    return df
