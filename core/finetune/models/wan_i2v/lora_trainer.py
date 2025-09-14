from ..wan_i2v.sft_trainer import WanI2VSftTrainer
from ..utils import register


class WanI2VLoraTrainer(WanI2VSftTrainer):
    pass


register("wan-i2v", "lora", WanI2VLoraTrainer)
