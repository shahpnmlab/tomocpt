from pathlib import Path
from typing import Union, Dict, Optional
from itertools import chain

from tomocpt.constants import RAW_LABEL_FNAME_TEMPLATE
from tqdm import tqdm
import functools
import math
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
def _paint_sphere(segmentation_vol: np.ndarray, point, radius_px: int, sphere: np.ndarray,
                  border_policy: str = "skip") -> bool:
    """Max-combine ``sphere`` (box ``2 * radius_px + 1``) centred at the integer point
    ``(x, y, z)``. ``"skip"`` (legacy) drops a particle whose box touches or crosses the
    border; ``"clip"`` draws its in-bounds part. Returns True if something was drawn."""
    tomo_dim = segmentation_vol.shape
    x, y, z = point
    zmin, zmax = int(z - radius_px), int(z + radius_px) + 1
    ymin, ymax = int(y - radius_px), int(y + radius_px) + 1
    xmin, xmax = int(x - radius_px), int(x + radius_px) + 1

    if (zmin <= 0 or zmax >= tomo_dim[0] or
            ymin <= 0 or ymax >= tomo_dim[1] or
            xmin <= 0 or xmax >= tomo_dim[2]):
        if str(getattr(border_policy, "value", border_policy)) != "clip":
            return False
        lo = [zmin, ymin, xmin]
        hi = [zmax, ymax, xmax]
        dst, src = [], []
        for a in range(3):
            l, h = max(0, lo[a]), min(tomo_dim[a], hi[a])
            if h <= l:
                return False
            dst.append(slice(l, h))
            src.append(slice(l - lo[a], h - lo[a]))
        dst, src = tuple(dst), tuple(src)
        segmentation_vol[dst] = np.maximum(segmentation_vol[dst], sphere[src])
        return True

    segmentation_vol[zmin:zmax, ymin:ymax, xmin:xmax] = np.maximum(
        segmentation_vol[zmin:zmax, ymin:ymax, xmin:xmax], sphere)
    return True


def _sphere_for_radius(radius_px: int, precision=np.float32) -> np.ndarray:
    sphere_box_size = int((2 * radius_px) + 1)
    if sphere_box_size % 2 == 0:
        sphere_box_size += 1
    return generate_gaussian_sphere(radius=radius_px, grid_size=sphere_box_size).astype(precision)


def create_particle_masks(vol_coord_pairs: Dict[str, np.ndarray],
                          radius_ang: float, tomogram_path: Union[str, Path],
                          output_path: Union[str, Path], class_id: str,
                          border_policy: str = "skip",
                          apix_override: Optional[float] = None,
                          radius_ang_per_point: Optional[Dict[str, np.ndarray]] = None):
    """
    Create particle masks for all coordinates.

    Args:
        vol_coord_pairs (Dict[str, np.ndarray]): Dictionary of coordinates per tomogram
        radius_ang (float): Radius in Angstroms
        tomogram_path (Union[str, Path]): Path to tomogram folder
        output_path (Union[str, Path]): Output path for masks
        class_id (str): Class identifier
        border_policy: "skip" (legacy) drops particles whose box touches the border,
            "clip" draws their in-bounds part
        apix_override: voxel size in A to use instead of the header value
        radius_ang_per_point: optional per-tomogram arrays of radii in A (multi-class labels);
            ``radius_ang`` is used when absent
    """
    precision = np.float32
    one_tomo_partial = sorted(list(vol_coord_pairs.keys()))[0]
    one_tomo_fname = list(Path(tomogram_path).glob(f"{one_tomo_partial}*.mrc"))[0]

    with mrcfile.open(one_tomo_fname) as mrc:
        tomo_dim = mrc.data.shape
        tomogram_pixel_size = mrc.voxel_size.x
    if apix_override is not None:
        tomogram_pixel_size = np.float32(apix_override)

    radius_px = round(radius_ang / tomogram_pixel_size)
    sphere = _sphere_for_radius(radius_px, precision)

    for filename, points in tqdm(vol_coord_pairs.items(), desc="Generating particle masks"):
        segmentation_vol = np.zeros(tomo_dim, dtype=precision)
        radii = None if radius_ang_per_point is None else radius_ang_per_point.get(filename)

        for k, point in enumerate(points):
            if radii is None:
                _paint_sphere(segmentation_vol, point, radius_px, sphere, border_policy)
            else:
                r_px = round(float(radii[k]) / tomogram_pixel_size)
                _paint_sphere(segmentation_vol, point, r_px, _sphere_for_radius(r_px, precision), border_policy)

        output_dir = Path(output_path) / f'class_{class_id}' / 'labels'
        output_dir.mkdir(parents=True, exist_ok=True)

        mrcfile.write(
            output_dir / f'{filename}.labels.mrc',
            segmentation_vol,
            voxel_size=tomogram_pixel_size,
            overwrite=True
        )


COORDS_COLUMNS = ["x", "y", "z", "class_number", "diameter_ang"]


def write_coords_csv(path: Union[str, Path], points_xyz, class_numbers, diameters_ang) -> Path:
    """``<dataset_dir>/coords/<tomo>.csv`` with columns ``x, y, z, class_number,
    diameter_ang`` (native voxels, the values used to draw the v1 labels)."""
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    n = len(pts)
    cls = np.broadcast_to(np.asarray(class_numbers, dtype=int), (n,)) if n else np.zeros((0,), int)
    dia = np.broadcast_to(np.asarray(diameters_ang, dtype=float), (n,)) if n else np.zeros((0,), float)
    df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
                       "class_number": cls.astype(int), "diameter_ang": dia.astype(float)},
                      columns=COORDS_COLUMNS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _tomo_file_for(tomo_path: Path, tomo_name: str) -> Optional[Path]:
    return next(chain(tomo_path.glob(f"*{tomo_name}*.mrc"), tomo_path.glob(f"*{tomo_name}*.rec")), None)


def _v2_tracking_columns(vol_coord_pairs, tomo_path: Path, output_dir: Path, class_numbers,
                         diameters_ang, apix_override: Optional[float], class_annotated: str) -> Dict[str, dict]:
    """Write one coordinate CSV per tomogram; return the extra tracking columns per tomogram
    (``coords_path``, ``apix_native``, ``class_annotated``)."""
    from tomocpt.dataManager.scaling import read_apix

    extra = {}
    for tomo_name, coordinates in vol_coord_pairs.items():
        tomo_file = _tomo_file_for(tomo_path, tomo_name)
        cls = class_numbers[tomo_name] if isinstance(class_numbers, dict) else class_numbers
        dia = diameters_ang[tomo_name] if isinstance(diameters_ang, dict) else diameters_ang
        coords_path = write_coords_csv(output_dir / "coords" / f"{tomo_name}.csv", coordinates, cls, dia)
        apix = read_apix(tomo_file, apix_override) if tomo_file is not None else float("nan")
        extra[tomo_name] = {"coords_path": str(coords_path), "apix_native": float(apix),
                            "class_annotated": str(class_annotated)}
    return extra


def prepare_picking_imod(input_file: Path ,
                         tomo_path: Path ,
                         output_dir: Path ,
                         particle_diameter_angst: float,
                         border_policy: str = "skip",
                         apix_override: Optional[float] = None,
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

    # Coordinate CSVs for pipeline v2 (class number 1, the dataset diameter)
    extra = _v2_tracking_columns(matched_data, tomo_path, output_dir, 1, particle_diameter_angst,
                                 apix_override, "1")

    # Create tracking DataFrame
    tracking_df = create_tracking_dataframe(matched_data, tomo_path, output_dir, particle_diameter_angst, class_id="all",
                                            extra_columns=extra)

    # Save tracking information
    tracking_csv = output_dir / "imod_picking_tracking.csv"
    tracking_df.to_csv(tracking_csv, index=False)
    logging.info(f"Saved tracking information to {tracking_csv}")

    # Create particle masks
    create_particle_masks(matched_data,
                          radius_ang=particle_radius_angst,
                          tomogram_path=tomo_path,
                          output_path=output_dir,
                          class_id="all",
                          border_policy=border_policy,
                          apix_override=apix_override)


def prepare_picking_star(input_file: Path,
                         tomo_path: Path,
                         output_dir: Path,
                         class_id: str,
                         particle_diameter_angst: float,
                         border_policy: str = "skip",
                         apix_override: Optional[float] = None,
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
    if apix_override is not None:
        tomogram_voxel_size = float(apix_override)
    vol_coord_pairs = star.get_shifted_scaled_coordinates(matched_data, tomogram_pixel_size=tomogram_voxel_size)

    # Coordinate CSVs for pipeline v2: class number from class_id ("all" -> 1), dataset diameter
    from tomocpt.labels.multiclass import single_class_number
    number = single_class_number(class_id)
    extra = _v2_tracking_columns(vol_coord_pairs, tomo_path, output_dir, number, particle_diameter_angst,
                                 apix_override, str(number))

    # Create tracking DataFrame
    tracking_df = create_tracking_dataframe(vol_coord_pairs, tomo_path, output_dir, particle_diameter_angst, class_id,
                                            extra_columns=extra)

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
        class_id=class_id,
        border_policy=border_policy,
        apix_override=apix_override,
    )


def prepare_picking_star_multiclass(input_file: Path,
                                    tomo_path: Path,
                                    output_dir: Path,
                                    class_table,
                                    class_annotated: Optional[str] = None,
                                    class_id_label: str = "all",
                                    border_policy: str = "skip",
                                    apix_override: Optional[float] = None,
                                    ) -> None:
    """Multi-class STAR data (``rlnClassNumber``): one dataset holds every species.

    Writes one label MRC per tomogram (every class drawn with its own radius, v1 rules), one
    coordinate CSV per tomogram with all classes, and a tracking CSV with the legacy columns
    plus ``coords_path``, ``apix_native`` and ``class_annotated``. A class in the STAR file
    without a diameter in the class table raises ValueError naming it.
    """
    from tomocpt.labels.multiclass import parse_class_annotated

    tomo_path = Path(tomo_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    star = Star(input_file)
    if "rlnClassNumber" not in star.particle_data.columns:
        raise ValueError(f"Multi-class data need rlnClassNumber in the STAR file {input_file}")
    diam = {int(r.number): float(r.diameter_ang) for r in class_table}
    present = sorted({int(c) for c in star.particle_data["rlnClassNumber"].unique()})
    missing = [c for c in present if c not in diam]
    if missing:
        raise ValueError(
            f"Class(es) {missing} in {input_file} have no diameter; add them to class_diameters")

    annotated = parse_class_annotated(class_annotated, class_table)
    if annotated is None:
        annotated = present
    annotated_txt = ",".join(str(c) for c in annotated)

    matched_data = match_data_to_tomograms(star.particle_data, tomo_path)
    tomogram_voxel_size = get_tomogram_pixel_size(tomo_path)
    if apix_override is not None:
        tomogram_voxel_size = float(apix_override)
    vol_coord_pairs, class_pairs = star.get_shifted_scaled_coordinates_with_classes(
        matched_data, tomogram_pixel_size=tomogram_voxel_size)
    diam_pairs = {k: np.array([diam[int(c)] for c in v], dtype=float) for k, v in class_pairs.items()}

    extra = _v2_tracking_columns(vol_coord_pairs, tomo_path, output_dir, class_pairs, diam_pairs,
                                 apix_override, annotated_txt)
    max_diameter = max(diam[c] for c in present) if present else max(diam.values())
    tracking_df = create_tracking_dataframe(vol_coord_pairs, tomo_path, output_dir, max_diameter, class_id_label,
                                            extra_columns=extra)
    tracking_csv = output_dir / "star_picking_tracking_multiclass.csv"
    tracking_df.to_csv(tracking_csv, index=False)
    logging.info(f"Saved tracking information to {tracking_csv}")

    create_particle_masks(
        vol_coord_pairs,
        radius_ang=max_diameter / 2.0,
        tomogram_path=tomo_path,
        output_path=output_dir,
        class_id=class_id_label,
        border_policy=border_policy,
        apix_override=apix_override,
        radius_ang_per_point={k: v / 2.0 for k, v in diam_pairs.items()},
    )

def create_tracking_dataframe(vol_coord_pairs, tomo_path: Path, output_dir: Path,
                              particle_diameter_angst: float, class_id:str,
                              extra_columns: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
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

        row = {
            'tomogram_path': str(tomo_file),
            'label_path': str(label_file),
            'particle_diameter_angst': particle_diameter_angst,
            'num_particles': len(coordinates)
        }
        if extra_columns and tomo_name in extra_columns:
            row.update(extra_columns[tomo_name])
        tracking_data.append(row)

    df = pd.DataFrame(tracking_data)
    return df


# ------------------------------------------------------------------------------------------
# Pipeline v2 targets (CLAUDE.md Sections 1.3 and 5.2)
# ------------------------------------------------------------------------------------------

def _profile(r: np.ndarray, radius: float) -> np.ndarray:
    """CLAUDE.md Section 1.3 profile with a float radius (max 1 at r = 0, 0 for r > R)."""
    ss = 0.33 * radius / 2.0
    sl = 0.66 * radius / 2.0
    r2 = r * r
    v = 0.5 * np.exp(-r2 / (2.0 * ss * ss)) + 0.5 * np.exp(-r2 / (2.0 * sl * sl))
    v[r > radius] = 0.0
    return v


def render_targets(points, out_shape_zyx, n_classes: int, border_policy: str = "clip") -> np.ndarray:
    """Dense soft targets ``[n_classes, *out_shape_zyx]`` (float32) from points.

    ``points``: ``[N, 5]`` = ``z, y, x, class_index, d_vox`` in voxels of the output array
    (float positions allowed). Each particle is drawn with the Section 1.3 profile, float
    radius ``R = d_vox / 2``, centred at its float position (not rounded), max-combined per
    channel. ``"clip"`` draws the in-bounds part; ``"skip"`` drops particles whose box
    touches the array border (the v1 rule).
    """
    shape = tuple(int(s) for s in out_shape_zyx)
    n_classes = int(n_classes)
    out = np.zeros((n_classes,) + shape, dtype=np.float32)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 5)
    if len(pts) == 0:
        return out
    policy = str(getattr(border_policy, "value", border_policy))
    if policy not in ("clip", "skip"):
        raise ValueError(f"border_policy must be 'clip' or 'skip', got '{border_policy}'")
    if policy == "skip":
        pts = pts[~touches_border(pts, shape)]
    for z, y, x, c, d in pts:
        R = float(d) / 2.0
        if not R > 0:
            continue
        ci = int(round(c))
        if not 0 <= ci < n_classes:
            raise ValueError(f"class index {c} outside [0, {n_classes})")
        centre = (z, y, x)
        lo = [max(0, int(math.floor(p - R))) for p in centre]
        hi = [min(s, int(math.floor(p + R)) + 1) for p, s in zip(centre, shape)]
        if any(h <= l for l, h in zip(lo, hi)):
            continue
        dz = np.arange(lo[0], hi[0], dtype=np.float64) - z
        dy = np.arange(lo[1], hi[1], dtype=np.float64) - y
        dx = np.arange(lo[2], hi[2], dtype=np.float64) - x
        r = np.sqrt(dz[:, None, None] ** 2 + dy[None, :, None] ** 2 + dx[None, None, :] ** 2)
        v = _profile(r, R).astype(np.float32)
        box = (ci, slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        np.maximum(out[box], v, out=out[box])
    return out


def touches_border(points, shape_zyx) -> np.ndarray:
    """Bool ``[N]``: the particle's box touches or crosses the array border (v1 rule:
    ``int(c - R) <= 0`` or ``int(c + R) + 1 >= dim`` on any axis, ``R = d_vox / 2``)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 5)
    if len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    R = pts[:, 4] / 2.0
    bad = np.zeros(len(pts), dtype=bool)
    for a, dim in enumerate(shape_zyx):
        lo = np.trunc(pts[:, a] - R)
        hi = np.trunc(pts[:, a] + R) + 1
        bad |= (lo <= 0) | (hi >= int(dim))
    return bad


def points_to_working(coords_df, apix_native: float, apix_work: float, class_table,
                      coord_map=None) -> np.ndarray:
    """Coordinates (``x, y, z, class_number`` in native voxels) -> ``[N, 5]`` float64 rows
    ``z, y, x, class_index, d_vox`` in working voxels.

    Without ``coord_map``, positions scale by ``apix_native / apix_work`` (voxel 0 at the
    origin). With ``coord_map = (scale_zyx, offset_zyx)`` (``scaling.coordinate_map`` of the
    resample that was run), position on axis ``a`` is ``c_a * scale_a + offset_a``. ``class_index`` is the class-table row of ``class_number``;
    ``d_vox = diameter_ang / apix_work`` from the class table. A class number that is not in
    the table raises ValueError.
    """
    if coords_df is None or len(coords_df) == 0:
        return np.zeros((0, 5), dtype=np.float64)
    index = {int(r.number): i for i, r in enumerate(class_table)}
    diam = [float(r.diameter_ang) for r in class_table]
    f = float(apix_native) / float(apix_work)
    numbers = np.asarray(coords_df["class_number"], dtype=np.int64)
    unknown = sorted({int(n) for n in numbers if int(n) not in index})
    if unknown:
        raise ValueError(f"Class number(s) {unknown} in the coordinates are not in the class table "
                         f"{sorted(index)}")
    ci = np.array([index[int(n)] for n in numbers], dtype=np.int64)
    out = np.zeros((len(coords_df), 5), dtype=np.float64)
    if coord_map is None:
        scale, offset = np.full(3, f), np.zeros(3)
    else:
        scale = np.asarray(coord_map[0], dtype=np.float64).reshape(3)
        offset = np.asarray(coord_map[1], dtype=np.float64).reshape(3)
    for a, col in enumerate(("z", "y", "x")):
        out[:, a] = np.asarray(coords_df[col], dtype=np.float64) * scale[a] + offset[a]
    out[:, 3] = ci
    out[:, 4] = np.array([diam[i] for i in ci], dtype=np.float64) / float(apix_work)
    return out
