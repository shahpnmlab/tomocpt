PACKAGE_NAME = "PYCOTOOL"

TRAIN_DIR_NAME = "train"
VAL_DIR_NAME = "val"
PYCO_MD_FILE_NAME = "pyco_md.csv"
VOLUMES_DIR_NAME_PREFIX = "volumes"
LABELS_DIR_NAME_PREFIX = "labels%s" #labels{selfSup|supervised}
PERCENT_TO_VALIDATE = 0.2
RAW_VOL_FNAME_PATTERN = r'([\w\-_]+)\.mrc'
RAW_LABEL_FNAME_TEMPLATE = "%s.labels.mrc"
CUBES_EXTENSION = ".nii.gz"
CUBES_FNAMES_TEMPLATES = "%s/%s_%d_%d_%d_%d" + CUBES_EXTENSION  #{VOLUMES_DIR_NAME_PREFIX/LABELS_DIR_NAME_PREFIX}/chunk_{ID}_[Z]_[Y]_[X]

#: Probability of flipping contrast polarity during training. Fixed rather than
#: exposed through config: invariance to whether particles are dark on light or
#: light on dark is wanted in every run, not something to tune per experiment.
CONTRAST_INVERSION_PROB = 0.5
