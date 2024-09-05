# 配置

该目录包含`ml_collections.ConfigDict`配置。
它的结构方式是将公共配置参数分解到`common.py`中，
将模型配置分解到`models.py`中。


要选择这些配置之一，您可以在命令行中指定:
```sh
# sh指脚本（shell）
python -m vit_jax.main --config=$(pwd)/vit_jax/configs/vit.py:b32,cifar10
```

上面的示例指定了附加参数`b32,cifar10`，该参数在文件`vitp .py`中解析，并使配置参数化。

注意，可以在命令行中通过指定诸如`--config.accumulation_steps=1`这样的附加参数来覆盖任何配置参数。