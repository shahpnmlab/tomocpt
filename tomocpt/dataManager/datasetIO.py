import os
import torchio as tio

from tomocpt import constants
from tomocpt.dataManager.dataUtils import get_labels_dirname


class VolumeDatsetIO(tio.SubjectsDataset):

    def _load(data_dir:str, return_labels:bool=True):
        lists_of_subjects = []
        

        for root, dirs, files in os.walk(os.path.join(data_dir), topdown=False): #TODO: possibly split train/test/val
            for name in files:
                if name.startswith(constants.VOLUMES_DIR_NAME_PREFIX) and name.endswith(constants.CUBES_EXTENSION):
                    realRoot, _ = os.path.split(root)
                    labelName = name.replace(constants.VOLUMES_DIR_NAME_PREFIX, get_labels_dirname(return_labels))
                    full_vol_name = os.path.abspath(os.path.join(realRoot, constants.VOLUMES_DIR_NAME_PREFIX, name))
                    full_label_name = os.path.abspath(os.path.join(realRoot, get_labels_dirname(return_labels), labelName))
                    assert os.path.isfile(full_vol_name)
                    if return_labels:
                        assert os.path.isfile(full_label_name)
                        subject = tio.Subject({
                            "input_data": tio.ScalarImage(full_vol_name),
                            "target_data": tio.LabelMap(full_label_name)
                        })
                    else:
                        subject = tio.Subject({
                            "input_data": tio.ScalarImage(full_vol_name),
                            "target_data": tio.LabelMap(full_vol_name)
                        })
                    lists_of_subjects.append(subject)
        return lists_of_subjects

    @staticmethod
    def get_dataset(data_dir:str, isTraining=True, load_getitem:bool=True, return_labels:bool=True):

        if isTraining:
            spatial = tio.OneOf({
                tio.RandomElasticDeformation(): 0.1,
                tio.RandomBlur(std=1):0.1
                #RandomAnisotropy
                #RandomFlip
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


