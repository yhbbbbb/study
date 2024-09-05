"""
Augreg(Augmentation and Regularization)
与卷积神经网络相比，当在较小的训练数据集上训练时，通常
发现Vision Transformer较弱的归纳偏差导致对模型正则化或数据增强(简称AugReg)

文档字符串，用于说明如何使用该脚本进行模型微调的示例和相关信息,具体作用如下：

功能说明：
Fine-tunes a Vision Transformer / Hybrid from AugReg checkpoint.
--表明代码的用途：用于从 AugReg 检查点微调一个 Vision Transformer（ViT）或 Hybrid 模型

使用示例：
Example for fine-tuning a R+Ti/16 on cifar100:

python -m vit_jax.main --workdir=/tmp/vit \
    --config=$(pwd)/vit_jax/configs/augreg.py:R_Ti_16 \
    --config.dataset=oxford_iiit_pet \
    --config.pp.train='train[:90%]' \
    --config.base_lr=0.01
--提供了一个微调R+Ti/16模型cifar100数据集山的命令行示例，指明如何运行脚本并传递配置参数

默认模型说明；
请注意，在默认情况下，根据上游验证精度，脚本将选择最佳的i21k预训练检查点。

手动选择模型：
用户可以通过指定完整的模型名称（不带 ".npz" 扩展名）来手动选择模型，
提供了如何通过命令行参数实现这一步的示例
python -m vit_jax.main --workdir=/tmp/vit \
    --config=$(pwd)/vit_jax/configs/augreg.py:R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0 \
    --config.dataset=oxford_iiit_pet \
    --config.pp.train='train[:90%]' \
    --config.base_lr=0.01
"""

import ml_collections  # 导入用于处理配置，导入其他模块以获取通用配置和模型配置

from vit_jax.configs import common
from vit_jax.configs import models


def get_config(model_or_filename):
  """返回在' dataset '上微调ViT ' model '的默认参数"""
  config = common.get_config()  # 从common模块获取基本的配置

  config.pretrained_dir = 'gs://vit_models/augreg'  # 指定预训练模型储存的位置

  config.model_or_filename = model_or_filename
  model = model_or_filename.split('-')[0]  # 将传入的模型文件名分解以提取模型名称将传入的模型文件名分解以提取模型名称

  if model not in models.AUGREG_CONFIGS:  # 检查传入模型的有效性
    raise ValueError(f'Unknown Augreg model "{model}"'
                     f'- not found in {set(models.AUGREG_CONFIGS.keys())}')
    # 设置模型配置
    # 使用从模型配置中提取的参数，并将 dropout rate 设置为 0，因为在微调期间不使用数据增
  config.model = models.AUGREG_CONFIGS[model].copy_and_resolve_references()
  config.model.transformer.dropout_rate = 0  # 微调期间没有AugReg

    # 设置其他超参数
    # These values are often overridden（覆盖） on the command line.
  config.base_lr = 0.03
  config.total_steps = 500
  config.warmup_steps = 100

    # 设置数据处理配置
    # 配置训练和测试数据的处理方式，包括图像的调整大小和裁剪
  config.pp = ml_collections.ConfigDict()
  config.pp.train = 'train'
  config.pp.test = 'test'
  config.pp.resize = 448
  config.pp.crop = 384

    # This value MUST be overridden on the command line.
  config.dataset = ''

    # 数据集名需要在命令行中覆盖，最后返回构建好的配置对象
  return config