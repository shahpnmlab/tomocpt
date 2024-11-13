import os
import torchio as tio

from tomocpt import constants
from tomocpt.dataManager.dataUtils import get_labels_dirname


class VolumeDatsetIO(tio.SubjectsDataset):

    @staticmethod
    def _load(data_dir: str, return_labels: bool = True):
        lists_of_subjects = []

        for root, dirs, files in os.walk(os.path.join(data_dir), topdown=False):  # TODO: possibly split train/test/val
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
                        subject = tio.Subject({
                            "input_data": tio.ScalarImage(full_vol_name),
                            "target_data": tio.ScalarImage(full_label_name)
                        })
                    else:
                        subject = tio.Subject({
                            "input_data": tio.ScalarImage(full_vol_name),
                            "target_data": tio.ScalarImage(full_vol_name)
                        })

                    lists_of_subjects.append(subject)
        return lists_of_subjects

    @staticmethod
    def get_dataset(data_dir: str, isTraining=True, load_getitem: bool = True, return_labels: bool = True):

        if isTraining:
            spatial = tio.OneOf({
                tio.RandomElasticDeformation(): 0.1,
                tio.RandomBlur(std=1): 0.1,
                tio.RandomElasticDeformation(
                    num_control_points=(7, 7, 7),
                    max_displacement=4,
                    locked_borders=2): 0.3,
                tio.RandomAffine(degrees=45,
                                 default_pad_value="otsu"): 0.8,
            }, p=0.75)

            noise = tio.OneOf({
                tio.RandomGhosting(num_ghosts=2): 0.2,
                tio.RandomBlur(std=(0.1, 1.5)): 0.3,
                tio.RandomBiasField(): 0.2,
            }, p=0.8)

            intensity = tio.OneOf({
                tio.RandomBlur(std=(0.1, 1.0)): 0.3
            })

            transform = tio.Compose([spatial, noise, intensity])
        else:
            transform = None
        listOfSubjects = VolumeDatsetIO._load(data_dir, return_labels=return_labels)
        assert listOfSubjects, f"Error, no valid listOfSubjects at data_dir {data_dir}"
        return VolumeDatsetIO(listOfSubjects, transform=transform, load_getitem=load_getitem)