from tomocpt.configManager.configManager import create_app
from tomocpt.mainConfig import mainConfig
from tomocpt.configManager.initializer import initialize_config
from tomocpt.labels.run import prepare_vol_label_pairs
from tomocpt.training.run import train
from tomocpt.infer.run import predict


def main():
    tomocpt_app = create_app()

    ##This is the proven way of doing it
    config_for_label_vol_preparation = mainConfig.prepData
    config_for_train = mainConfig.train
    config_for_prediction = mainConfig.infer

    tomocpt_app.register_command(initialize_config, None) #TODO: init_config needs to be modified to generate a config file for train, another for inference and so one
    tomocpt_app.register_command(prepare_vol_label_pairs, config_for_label_vol_preparation)
    tomocpt_app.register_command(train, config_for_train)
    tomocpt_app.register_command(predict, config_for_prediction)
    tomocpt_app.run()

if __name__ == "__main__":
    main()
    """

python -m tomocpt.main train --network.model_type unet --chunks_dir data/refactor/chunks/ --model_dir /tmp/unet --n_epochs 2 OVERFIT_N_BATCHES=10
python -m tomocpt.main train --network.model_type unet --chunks_dir data/refactor/chunks/ --model_dir /tmp/unet   --n_epochs 2 OVERFIT_N_BATCHES=10 network.KERNEL_SIZE=7
python -m tomocpt.main train --config_file externalConfExamples/trainExternalConf.yaml 

python -m tomocpt.main infer --tomosDir data/refactor/datasets/particle1/tomograms/ --predsDir /tmp/kk_inference --modelFname /tmp/unet/unnamed/checkpoints/weights.ckpt  --particleLengthAng 250

    """