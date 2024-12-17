from tomocpt.configManager.configManager import create_app
from tomocpt.mainConfig import mainConfig
from tomocpt.configManager.initializer import init
from tomocpt.labels.run import prepare_labels
from tomocpt.training.run import train
from tomocpt.infer.run import infer


app = create_app()


config_for_train = mainConfig.train
config_for_inference = mainConfig.infer
config_for_label_vol_preparation = mainConfig.prepData

app.register_command(init, None) #TODO: init_config needs to be modified to generate a config file for train, another for inference and so one
app.register_command(prepare_labels, config_for_label_vol_preparation)
app.register_command(train, config_for_train)
app.register_command(infer, config_for_inference)

if __name__ == "__main__":
    app.run()
    """

python -m tomocpt.main train --network.model_type unet --chunks_dir data/refactor/chunks/ --model_dir /tmp/unet --n_epochs 2 OVERFIT_N_BATCHES=10
python -m tomocpt.main train --network.model_type unet --chunks_dir data/refactor/chunks/ --model_dir /tmp/unet   --n_epochs 2 OVERFIT_N_BATCHES=10 network.KERNEL_SIZE=7
python -m tomocpt.main train --config_file externalConfExamples/trainExternalConf.yaml 

python -m tomocpt.main infer --tomosDir data/refactor/datasets/particle1/tomograms/ --predsDir /tmp/kk_inference --modelFname /tmp/unet/unnamed/checkpoints/weights.ckpt  --particleLengthAng 250

    """