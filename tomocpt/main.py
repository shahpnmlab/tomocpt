from tomocpt.configManager.configManager import create_app
from tomocpt.mainConfig import mainConfig
from tomocpt.configManager.initializer import initialize_config
from tomocpt.labels.run import prepare_vol_label_pairs
from tomocpt.training.run import train
from tomocpt.infer.run import predict

def main():
    tomocpt_app = create_app()

    tomocpt_app.register_command(initialize_config, mainConfig, None, False)
    tomocpt_app.register_command(prepare_vol_label_pairs, mainConfig, "prepData")
    tomocpt_app.register_command(train, mainConfig, "train")
    tomocpt_app.register_command(predict, mainConfig, "infer")
    tomocpt_app.run()

if __name__ == "__main__":
    main()
    """

python -m tomocpt.main train --network.model_type unet --chunks_dir data/refactor/chunks/ --model_dir /tmp/unet --n_epochs 2 OVERFIT_N_BATCHES=10
python -m tomocpt.main train --network.model_type unet --chunks_dir data/refactor/chunks/ --model_dir /tmp/unet   --n_epochs 2 OVERFIT_N_BATCHES=10 network.KERNEL_SIZE=7
python -m tomocpt.main train --config_file externalConfExamples/trainExternalConf.yaml 

python -m tomocpt.main infer --tomosDir data/refactor/datasets/particle1/tomograms/ --predsDir /tmp/kk_inference --modelFname /tmp/unet/unnamed/checkpoints/weights.ckpt  --particleLengthAng 250

    """