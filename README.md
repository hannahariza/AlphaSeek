本文档提供了更新后 **QuantaAlpha** 与 **RD-Agent** 两个项目的核心实现步骤与环境配置说明。

---

## 1. QuantaAlpha 复现说明

对于 QuantaAlpha 的所有复现工作，请务必以 **[官方原项目文档](https://github.com/QuantaAlpha/QuantaAlpha?tab=readme-ov-file)** 为基准。

### 📌 核心配置与运行要求

* **环境与数据**：关于基础环境的搭配、如何通过 `qlib` 下载股票数据以及基础的运行指令，均请严格参考原项目 README。
* **大模型 API 配置**：相关大模型（LLM）的密钥必须由用户自行在项目根目录的 `.env` 文件中完成配置（原项目文档中有详细配置说明）。
* **因子挖掘与回测**：一切配置就绪后，你可以直接通过**前端**或**后端**启动因子挖掘与单因子指标回测任务。

> **⚠️ 注意事项**：使用前端运行和后端运行产生的回测结果，其默认保存路径可能有所不同，请在运行后留意文件输出位置。

---

## 2. RD-Agent 部署与运行

对于 RD-Agent 项目，**优先推荐使用 Docker 容器**进行部署以保证环境一致性。详细的 Docker 配置请查阅 **[RD-Agent 官方文档](https://github.com/microsoft/RD-Agent)**。

如果您更倾向于使用本地的 Conda 虚拟环境，请按照以下步骤进行操作：

### 🛠️ Conda 环境配置步骤

```bash
# 1. 进入项目根目录
cd RD-Agent

# 2. 创建并激活 Python 3.10 的虚拟环境
conda create -n rdagent python=3.10
conda activate rdagent

# 3. 以开发者（可编辑）模式安装项目包（强制指定版本号以绕过 Git 校验）
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e .

# 4. 安装其余必要依赖
pip install -r requirements.txt

🚀 运行迭代任务
环境准备完毕后，可以使用以下命令启动模型迭代与组合迭代：

Bash
python -m rdagent.app.qlib_rd_loop.quant_v2 --loop_n 10
参数说明：

rdagent.app.qlib_rd_loop.quant_v2：此为该路径下的主运行文件。

--loop_n 10：指定迭代的总次数为 10 次。您可以根据实际计算资源和需求自行修改该数值。
