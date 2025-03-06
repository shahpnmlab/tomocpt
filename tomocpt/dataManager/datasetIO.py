import os
import torchio as tio
from joblib import Parallel, delayed

from tomocpt import constants
from tomocpt.dataManager.dataUtils import get_labels_dirname
from tomocpt.mainConfig import mainConfig

class ScalarImage(tio.ScalarImage):
    def _check_data_loader(self):
        return

class LabelMap(tio.LabelMap):
    def _check_data_loader(self):
        return

class VolumeDatsetIO(tio.SubjectsDataset):

    @staticmethod
    def _load(data_dir: str, return_labels: bool = True):

        names_list = []
        for root, dirs, files in os.walk(os.path.join(data_dir), topdown=False):
            for name in files:
                if name.startswith(constants.VOLUMES_DIR_NAME_PREFIX) and name.endswith(constants.CUBES_EXTENSION):
                    realRoot, _ = os.path.split(root)
                    labelName = name.replace(constants.VOLUMES_DIR_NAME_PREFIX, get_labels_dirname(return_labels))
                    full_vol_name = os.path.abspath(os.path.join(realRoot, constants.VOLUMES_DIR_NAME_PREFIX, name))
                    full_label_name = os.path.abspath(
                        os.path.join(realRoot, get_labels_dirname(return_labels), labelName))
                    assert os.path.isfile(full_vol_name)
                    if return_labels:
                        assert os.path.isfile(full_label_name)
                        names_list.append((full_vol_name, full_label_name))
                    else:
                        names_list.append((full_vol_name, full_vol_name))

        def create_subject(x, y):
            subject = tio.Subject({
                "input_data": ScalarImage(x),
                "target_data": LabelMap(y)
            })
            return subject
        # print("mainConfig.train.WORKERS_FOR_DATA",mainConfig.train.WORKERS_FOR_DATA, flush=True)
        n_jobs = mainConfig.train.n_cpus_for_train
        if n_jobs == 0:
            n_jobs = 1
        lists_of_subjects = Parallel(n_jobs=n_jobs)(delayed(create_subject)(x, y) for x,y in names_list)
        return lists_of_subjects

    @staticmethod
    def get_dataset(data_dir: str, isTraining=True, load_getitem: bool = True, return_labels: bool = True):

        if isTraining:
            spatial = tio.OneOf({
                tio.RandomElasticDeformation(): 0.1,
                tio.RandomBlur(std=1): 0.1
                # RandomAnisotropy
                # RandomFlip
            }, p=0.75)

            transform = tio.Compose([
                tio.RandomAffine(degrees=45, default_pad_value="otsu", p=0.8),
                spatial
            ])
        else:
            transform = None
        listOfSubjects = VolumeDatsetIO._load(data_dir, return_labels=return_labels)
        assert listOfSubjects, f"Error, no valid listOfSubjects at data_dir {data_dir}"
        return VolumeDatsetIO(listOfSubjects, transform=transform, load_getitem=load_getitem)
