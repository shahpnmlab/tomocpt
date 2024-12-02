import typer
from omegaconf import DictConfig

from tomocpt.defaultConfigs.infer_config import OutputFormat

try:
    from itertools import batched
except:
    from more_itertools import batched

from typing import Annotated

from joblib import Parallel, delayed

from tomocpt.infer.helpers import * #TODO: * imports are not a good practise
import logging


def infer(plot: Annotated[bool, typer.Option(help="#TODO")] = False, #TODO: should this be removed?
          config: DictConfig = None):

    from tomocpt.mainConfig import mainConfig
    infer_config = mainConfig.infer
    from tomocpt.utils import accelerator_selector


    tomosDirPath = Path(infer_config.tomosDir).resolve()

    Path(infer_config.predsDir).mkdir(parents=True, exist_ok=True)
    data_fnames = sorted(list(tomosDirPath.glob('*.mrc')))
    accel, n_gpus = accelerator_selector(use_cuda=infer_config.use_cuda, n_cpus=infer_config.N_CPUS_IF_NO_GPU)
    results = Parallel(n_jobs=infer_config.oversubscribeFactor * n_gpus,
                       batch_size=1)(delayed(infer_tomos)(batch_fnames, infer_config.predsDir, infer_config.modelFname,
                                                          particleLengthAng=infer_config.particleLengthAng,
                                                          gpu_id=(i % infer_config.oversubscribeFactor) % n_gpus,
                                                          batch_size=infer_config.batch_size, plot=plot,
                                                          save_pred_mask=infer_config.savePredMasks,
                                                          extract_coords=infer_config.extractCoords,
                                                          nearest_neigs_angs=infer_config.nearest_neigs_angs,
                                                          threshold=infer_config.deep_threshold,
                                                          masksDir=infer_config.masksDir)
                                     for i, batch_fnames in
                                     enumerate(batched(data_fnames, n=infer_config.oversubscribeFactor * n_gpus)))

    if infer_config.extractCoords:
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

        if mainConfig.infer.outCoordFormat == OutputFormat.relion:
            write_relion_star_file(output_dir=mainConfig.infer.predsDir,
                                       tomo_names=tomoNames,
                                       predicted_centroids_with_scores=predicted_centroids_with_scores,
                                       voxel_size = voxel_sizes, #TODO: voxel_sizes is a list of voxel_sizes!!
                                       output_filename = mainConfig.infer.outCoordFname,
                                       version= 3.1) #TODO: put this into config
        else:
            raise NotImplementedError()
        logging.info(f"Predicted coordinates are stored here: {mainConfig.infer.outCoordFname}") #TODO: print the absolute path instead


MODEL = None
def infer_tomos(tomoFnames: List[Path], predsDir: str, modelFname: str, gpu_id: int, particleLengthAng: float,
                batch_size: int,
                plot: bool, save_pred_mask: bool, extract_coords: bool,
                nearest_neigs_angs: Optional[float], threshold: float, masksDir: Optional[str]):

    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.infer.helpers import _infer_one_tomo

    global MODEL
    kwargs = {"map_location": f"cuda:{gpu_id}"}
    if MODEL is None:
        MODEL = BasePickingModel.load_from_checkpoint(modelFname, **kwargs)

    model = MODEL.eval().cuda(gpu_id)
    patch_size = model.patch_size

    tomo_names, one_tomo_centroids_and_scores, voxel_sizes = [], [], []
    for data_fname in tomoFnames:
        output_fname = str(Path(predsDir) / data_fname.name)
        mask_fname = Path(masksDir) / data_fname.name if masksDir else None
        _tomo_names, _one_tomo_centroids_and_scores, vx = _infer_one_tomo(tomoFname=str(data_fname),
                                                                          output_fname=output_fname,
                                                                          particle_ang_length=particleLengthAng,
                                                                          model=model, gpu_id=gpu_id,
                                                                          patch_size=patch_size, batch_size=batch_size,
                                                                          plot=plot, save_pred_mask=save_pred_mask,
                                                                          extract_coords=extract_coords,
                                                                          nearest_neigs_angs=nearest_neigs_angs,
                                                                          threshold=threshold,
                                                                          maskFname=str(mask_fname) if mask_fname else None)
        tomo_names.extend(_tomo_names)
        one_tomo_centroids_and_scores.extend(_one_tomo_centroids_and_scores)
        voxel_sizes.append(vx)
    return tomo_names, one_tomo_centroids_and_scores, voxel_sizes
