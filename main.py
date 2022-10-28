import os
from config import opt
from trainer import setup_seed
from trainer import CombineTrainer
import warnings

# ignore warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if __name__ == "__main__":
    setup_seed(opt.seed)
    trainer = CombineTrainer(opt)
    trainer.train()