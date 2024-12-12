import gc
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Literal, Optional, Union

import torch
import torchio as tio
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from tomocpt import constants
from tomocpt.dataManager.dataUtils import load_mrc, resize_volume
from tomocpt.dataPreparation.helpers import _preprocess_data_mrc, get_labels_dirname, get_vol_chunks
from tomocpt.mainConfig import mainConfig
from tomocpt.utils import accelerator_selector, makedir


def process_mrc(data_fname, target_fname, particle_diameter_angst,
                outputname_template: Optional[str],
                chunk_size: Optional[int]=None, stride: Optional[int]=None,
                normalization_function: str = "robust_normalization",
                new_size: Optional[int]=None, require_labels: bool = True):
    '''
    F0 = Compute number of zeros in vol => (torch.isclose(target,0).sum()
    F1 = Compute number of non-zeros in vol => target.numel() - F0
    Drop probability = 1-(F1/F0)
    Computes the drop probability for a given ratio of the number of examples in two classes F1 and F0, and a scalar
    parameter alpha.
    If alpha is 0, drop all examples in the class with fewer samples.
    If alpha is 1, drop as many examples as the number of samples in the minority class, so that the two classes are
    balanced.
    If alpha is 2, drop twice as many examples from the minority class as the number of samples in that class.
    Args:
    alpha (float): a scalar parameter that controls the severity of the drop.
    F1 (int): the number of examples in the minority class.
    F0 (int): the number of examples in the majority class.
    Returns:
    drop_probability (tensor): a tensor of the same shape as F1 and F0, containing the drop probabilities for each
    class.
    '''
    particle_radius_angst = particle_diameter_angst * .5
    if outputname_template is None:
        outputname_template = constants.CUBES_FNAMES_TEMPLATES

    if chunk_size is None:
        chunk_size = mainConfig.train.CHUNK_SIZE

    if stride is None:
        stride = mainConfig.train.CHUNK_STRIDE

    if new_size is None:
        new_size = mainConfig.prepData.desired_particle_pixel_size

    use_gpu = mainConfig.prepData.USE_CUDA_FOR_DATA
    vol, new_shape, old_shape, voxel_size, padding_values, scalar = _preprocess_data_mrc(data_fname,
                                                                                              particle_radius_angst,
                                                                                              normalization_function,
                                                                                              new_size,
                                                                                              chunk_size,
                                                                                              use_gpu)
    accel, _ = accelerator_selector(use_cuda=use_gpu)

    # drop_probablity = constants.PROBABILITY_OF_EMPTY_TO_DROP #TODO: change this parameter in config for alpha, defined below
    alpha =  mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES

    drop_probablity = 0.
    if target_fname:
        vol_target = load_mrc(target_fname, normalize=None, return_boxSize=False)
        vol_target = torch.tensor(vol_target)
        F0 = torch.isclose(vol_target, torch.zeros_like(vol_target)).sum()
        F1 = torch.numel(vol_target) - F0
        drop_probablity = torch.clip(1 - alpha * (F1 / F0), torch.zeros(1), torch.ones(1))
    else:
        vol_target = torch.zeros_like(vol)

    if accel == 'gpu':
        vol_target = vol_target.cuda()
        vol = vol.cuda()

    if vol_target.shape != tuple(new_shape):
        assert min(new_shape) >= chunk_size
        vol_target, _ = resize_volume(volume=vol_target, new_shape=new_shape, chunk_size=chunk_size, use_gpu=use_gpu)

    n_cubes = 0
    for i, ((chunk_data, chunk_target), coords) in enumerate(
            get_vol_chunks(vol, vol_target, chunk_size, stride=stride)):

        if chunk_target.sum() <= 0:
            if random.random() < drop_probablity:
                continue

        chunk_data_fname = outputname_template % (constants.VOLUMES_DIR_NAME_PREFIX,
                                                  constants.VOLUMES_DIR_NAME_PREFIX,
                                                  i, *tuple(coords))
        # chunk_data = tio.ScalarImage(tensor=torch.unsqueeze(chunk_data, 0).cpu())
        chunk_data = tio.ScalarImage(tensor=torch.Tensor(chunk_data[None, ...]).cpu())
        chunk_data.save(chunk_data_fname, squeeze=False)
        del chunk_data
        gc.collect()

        if target_fname:
            labels_dir_prefix = get_labels_dirname(require_labels)
            chunk_target_fname = outputname_template % (labels_dir_prefix,
                                                        labels_dir_prefix,
                                                        i, *tuple(coords))
            chunk_target = tio.ScalarImage(tensor=torch.Tensor(chunk_target[None, ...]).cpu())
            # chunk_target = tio.LabelMap(tensor=torch.unsqueeze(chunk_target, 0).cpu())
            chunk_target.save(chunk_target_fname, squeeze=False)
            del chunk_target
            gc.collect()
        n_cubes += 1
    return data_fname, n_cubes


# def _prep_self_supervised(particle_size: float, source_path: Union[str, Path],
#                           destination_path: str,
#                           class_id: str = 'all'):
#     """Creates a symbolic link from source path to destination path.
#
#     Args:
#         particle_size: size of the target particle in Angstroms
#         source_path: The path to the source file or directory.
#         destination_path: The path to the destination symlink.
#         class_id: which classes to run this training on.
#     Returns:
#       None.
#     """
#
#     # Check if the source path exists.
#     if not Path(source_path).exists():
#         raise FileNotFoundError(f"The source path {source_path} does not exist.")
#
#     # Check if the destination path exists.
#     if Path(destination_path).exists():
#         shutil.rmtree(destination_path, ignore_errors=True)
#
#     # Check if the destination path exists.
#     if not Path(f'{destination_path}/class_{class_id}/{constants.VOLUMES_DIR_NAME_PREFIX}').exists():
#         Path(f'{destination_path}/class_{class_id}/{constants.VOLUMES_DIR_NAME_PREFIX}').mkdir(parents=True,
#                                                                                                exist_ok=True)
#     # Create the symlink.
#     for file in Path(source_path).resolve().rglob("*.mrc"):
#         Path(
#             f'{destination_path}/class_{class_id}/{constants.VOLUMES_DIR_NAME_PREFIX}/{file.name}').resolve().symlink_to(
#             file)
#
#     path_to_csv_md = f'{destination_path}/class_{class_id}'
#     _, tomo_voxel = get_tomo_dims(source_path)
#     particle_size_px = ang_to_pix(particle_size, tomo_voxel)
#     write_md_csv(class_id='all', particle_size=particle_size_px, sampling_rate=tomo_voxel,
#                  parent_dir=path_to_csv_md, selfSupervised=True)
#
#
# def prepare_self_supervised_wrapper(tomoDirsPath: str = config.RAW_DATA_DIR,
#                                     annotatedDir: str = config.PREPARED_DATA_DIR):
#     """
#     :param tomoDirsPath:The path to the directory containing the datasets.
#     :param annotatedDir: The path to where the inputs for selfSupervised learning should be stored.
#     :return:
#     """
#     tomoDirsPath = os.path.expanduser(tomoDirsPath)
#     annotatedDir = os.path.expanduser(annotatedDir)
#
#     tomoDirs = sorted(list(Path(tomoDirsPath).glob("*")))
#     for tomoDir in tomoDirs:
#         if tomoDir.is_dir():
#             destination_path = f'{annotatedDir}/{tomoDir.name}'
#             particle_size_angst_path = f'{tomoDir}/{constants.PYCO_MD_FILE_NAME}'
#             if not Path(particle_size_angst_path).exists():
#                 warnings.warn(f"{particle_size_angst_path} does not exist. Skipping dir.")
#             else:
#                 particle_radius_angst, _, _ = read_md_csv(particle_size_angst_path)
#                 _prep_self_supervised(particle_size=particle_radius_angst, source_path=tomoDir,
#                                       destination_path=destination_path)


def do_chunking(tomosDf: str, chunkedDataDir,
                 n_cpus: int,
                 require_labels: bool = True,
                 train_val_level="tomos"):
    """

    :param tomosDf: This is the source of mrc files and potential labels
    :param chunkedDataDir: This is where chunked cubes will be stored
    :param n_cpus: The number of cpus to use
    :param require_labels: Whether the data is for supervised training and thus requires labels
    :return:
    """

    if train_val_level == "tomos":
        df_train, df_val = train_test_split(tomosDf, test_size=constants.PERCENT_TO_VALIDATE)
    else:
        raise NotImplementedError()

    n_cpus = 1 if n_cpus == 0 else n_cpus
    train_outDir = f'{chunkedDataDir}/{constants.TRAIN_DIR_NAME}'  # path/to/training/
    val_outDir = f'{chunkedDataDir}/{constants.VAL_DIR_NAME}'  # path/to/val/

    if os.path.isdir(train_outDir):
        shutil.rmtree(train_outDir, ignore_errors=False)
    if os.path.isdir(val_outDir):
        shutil.rmtree(val_outDir, ignore_errors=False)

    outputname_template = constants.CUBES_FNAMES_TEMPLATES

    def dispatcher(i, info_row, outpath):
        bn = Path(info_row["tomogram_path"]).stem
        bn_fname = f'{outpath}/{bn}/{outputname_template}'
        makedir(f'{outpath}/{bn}/{constants.VOLUMES_DIR_NAME_PREFIX}')

        if require_labels:
            labels_names_prefix = get_labels_dirname(True)
        else:
            labels_names_prefix = get_labels_dirname(False)

        makedir(f'{outpath}/{bn}/{labels_names_prefix}')
        if require_labels:
            target_fname = info_row["label_path"]
        else:
            target_fname = None
        particle_diameter_angst = info_row["particle_diameter_angst"]
        data_fname, n_cubes = process_mrc(data_fname=info_row["tomogram_path"], target_fname=target_fname,
                                          particle_diameter_angst=particle_diameter_angst,
                                          outputname_template=bn_fname, require_labels=require_labels)

        print(f"{data_fname}: {n_cubes} cubes written")



    Parallel(n_cpus)(delayed(dispatcher)(i, trainObj, train_outDir) for i, trainObj in df_train.iterrows())
    Parallel(n_cpus)(delayed(dispatcher)(i, valObj, val_outDir) for i, valObj in df_val.iterrows())
    print(f"Prepared data saved at: \n{train_outDir}\n{val_outDir}")
    with open(f'{chunkedDataDir}/done.txt', "w") as f:
        f.write("done")

