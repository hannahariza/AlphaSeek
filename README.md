对于一切QuantaAlpha的复现，一切参考
https://github.com/QuantaAlpha/QuantaAlpha?tab=readme-ov-file
原项目文件，包括环境的搭配，通过qlib进行股票数据的下载，运行指令，注：相关大模型的密钥需要自己在.env中进行配置（原QuantaAlpha项目文档中已说明）。配置好之后可以直接在可以在前端或者后端进行因子挖掘与单因子指标回测，但是保存路径可能不同。

对于RD-Agent，优先推荐使用docker容器，详细配置见源文档https://github.com/microsoft/RD-Agent
或者使用conda虚拟环境：
cd RD-Agent
conda create -n rdagent python=3.10
conda activate rdagent
# Install the package in development mode
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e .

# Install additional dependencies
pip install -r requirements.txt

进行模型迭代与组合迭代的命令：
python -m rdagent.app.qlib_rd_loop.quant_v2 --loop_n 10
其中rdagent.app.qlib_rd_loop.quant_v2是路径下的主运行文件；10是迭代次数，可以自己设定
