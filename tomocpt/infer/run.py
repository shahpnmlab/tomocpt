import typer
try:
    from itertools import batched
except:
    from more_itertools import batched

from pathlib import Path
from omegaconf import DictConfig
from typing import Annotated
from joblib import Parallel, delayed
from tomocpt.infer.helpers import  process_extracted_coordinates, infer_tomos



def infer(plot: Annotated[bool, typer.Option(help="Plot the cubes")] = False,
          config: DictConfig = None):
    from tomocpt.mainConfig import mainConfig
    from tomocpt.utils import accelerator_selector
    infer_config = mainConfig.infer

    tomosDirPath = Path(infer_config.tomosDir).resolve()
    Path(infer_config.predsDir).mkdir(parents=True, exist_ok=True)

    data_fnames = sorted(list(tomosDirPath.glob('*.mrc')))
    accel, n_gpus = accelerator_selector(use_cuda=infer_config.use_cuda, n_cpus=infer_config.N_CPUS_IF_NO_GPU)

    # Run parallel inference
    results = Parallel(n_jobs=infer_config.oversubscribeFactor * n_gpus, batch_size=1)(
        delayed(infer_tomos)(
            batch_fnames,
            infer_config.predsDir,
            infer_config.modelFname,
            particleLengthAng=infer_config.particleLengthAng,
            gpu_id=(i % infer_config.oversubscribeFactor) % n_gpus,
            batch_size=infer_config.batch_size,
            plot=plot,
            save_pred_mask=infer_config.savePredMasks,
            extract_coords=infer_config.extractCoords,
            nearest_neigs_angs=infer_config.nearest_neigs_angs,
            threshold=infer_config.deep_threshold,
            masksDir=infer_config.masksDir
        )
        for i, batch_fnames in enumerate(batched(data_fnames, n=infer_config.oversubscribeFactor * n_gpus))
    )

    if infer_config.extractCoords:
        process_extracted_coordinates(
            results=results,
            output_dir=mainConfig.infer.predsDir,
            output_format=mainConfig.infer.outCoordFormat,
            output_filename=mainConfig.infer.outCoordFname
        )



