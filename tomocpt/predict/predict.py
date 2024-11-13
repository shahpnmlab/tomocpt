from itertools import batched
from typing import Annotated, Optional

import typer
from joblib import Parallel, delayed

from tomocpt import constants
from tomocpt.mainConfig import mainConfig
from tomocpt.utils import accelerator_selector

network_config = mainConfig.network
train_config = mainConfig.train
infer_config = mainConfig.infer

from tomocpt.predict.helpers import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def infer(tomosDir: Annotated[Path, typer.Option(help="The directory that contains the tomograms with the same particle size")] ,
          predsDir: Annotated[Path, typer.Option(help="The directory where predictions will be saved")] ,
          modelFname: Annotated[Path, typer.Option(help="The model fname")],
          particleLengthAng: Annotated[float, typer.Option(help="Particle diameter in Angstroms")], #TODO: Pranav, check this
          batch_size: Annotated[int, typer.Option(help="batch size")]=infer_config.batch_size,
          oversubscribeFactor: int = 1,
          plot: bool = False,
          savePreds: bool = False,
          extractCoords: bool = True,
          nearest_neigs_angs: Optional[float] = None,
          deep_threshold: float = 0.3,
          outCoordFname: str = "tomopicker_coords.star",
          masksDir: Annotated[Optional[Path], typer.Option(help="The directory with masks")] = None
          ):
    """
    :param tomosDir: path to folder containing tomograms
    :param predsDir: path to directory to store inferred output
    :param modelFname: Path to trained model weights
    :param particleLengthAng: The length of the particle along its longest axis (Å)
    :param patch_size: Size of chunk to run inference on
    :param batch_size: batch size. Number of chunks to process together in the GPU
    :param oversubscribeFactor: The number of tomographs per-gpu to be processed in parallel.
    :param plot: plot raw inference_data and segmentation mask for quick viz.
    :param savePreds: whether to save the predicted segmentation mask
    :param extractCoords: whether to extract coordinates from the predicted segmentation mask
    :param nearest_neigs_angs: nearest neighbor distance (Å)
    :param deep_threshold: threshold for peak detection
    :param outCoordFname: output filename for coordinates
    :param masksDir: path to folder containing mask files (optional)
    :return:
    """


    tomosDirPath = Path(tomosDir).resolve()

    Path(predsDir).mkdir(parents=True, exist_ok=True)
    data_fnames = sorted(list(tomosDirPath.glob('*.mrc')))

    accel, n_gpus = accelerator_selector(use_cuda=infer_config.use_cuda, n_cpus=infer_config.N_CPUS_IF_NO_GPU)
    #TODO: patch size needs to be read from the model
    results = Parallel(n_jobs=oversubscribeFactor * n_gpus,
                       batch_size=1)(delayed(infer_tomos)(batch_fnames, predsDir, modelFname,
                                                          particleLengthAng=particleLengthAng,
                                                          gpu_id=(i % oversubscribeFactor) % n_gpus,
                                                          batch_size=batch_size, plot=plot,
                                                          save_preds=savePreds,
                                                          extract_coords=extractCoords,
                                                          nearest_neigs_angs=nearest_neigs_angs, threshold=deep_threshold,
                                                          masksDir=masksDir)
                                     for i, batch_fnames in
                                     enumerate(batched(data_fnames, oversubscribeFactor * n_gpus)))

    if extractCoords:
        tomoNames = []
        predicted_centroids_with_scores = []
        voxel_sizes = []
        # Unpack the results
        for res in results:
            tomoNames.extend(res[0])
            predicted_centroids_with_scores.extend(res[1])
            voxel_sizes.extend([res[2]] * len(res[0]))

        all_tomo_centroids_and_scores = {"rlnMicrographName": [],
                                         "rlnCoordinateX": [],
                                         "rlnCoordinateY": [],
                                         "rlnCoordinateZ": [],
                                         "rlnAutopickFigureOfMerit": []
                                         }

        for tomoName, predicted_centroid in zip(tomoNames, predicted_centroids_with_scores):
            tomoName = re.sub(r'_\d+\.\d+Apx', '.tomostar', tomoName)
            all_tomo_centroids_and_scores["rlnMicrographName"].append(tomoName)
            all_tomo_centroids_and_scores["rlnCoordinateX"].append(predicted_centroid[2])
            all_tomo_centroids_and_scores["rlnCoordinateY"].append(predicted_centroid[1])
            all_tomo_centroids_and_scores["rlnCoordinateZ"].append(predicted_centroid[0])
            all_tomo_centroids_and_scores["rlnAutopickFigureOfMerit"].append(predicted_centroid[3])

        df_optics = pd.DataFrame({ #TODO: Why do you invent this optical group?
            'rlnOpticsGroup': [1],
            "rlnOpticsGroupName": ["OpticsGroup1"],
            'rlnSphericalAberration': [2.7],
            'rlnVoltage': [300],
            'rlnImagePixelSize': [voxel_sizes[0]],
            'rlnImageDimensionality': [3]
        })
        df_particles = pd.DataFrame(data=all_tomo_centroids_and_scores)

        star_data = {
            #'optics': df_optics, #TODO: Pranav, check why you don't include the optical group
            'particles': df_particles
        }
        star_out = Path(f"{predsDir}/{outCoordFname}")
        starfile.write(star_data, star_out, float_format="%0.2f", overwrite=True)
        logger.info(f"Predicted coordinates are stored here: {star_out}")