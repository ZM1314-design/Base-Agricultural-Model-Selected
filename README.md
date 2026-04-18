# 基座模型

农业领域大语言模型的 **基座** 工程：用于 **继续预训练（CPT）**、**指令微调（SFT）**（含 **LoRA**）以及本地推理与部署。本仓库提供目录约定、依赖与最小运行脚本；**不包含**模型权重本身，可前往对应网站下载模型权重（见下表）。

---

## 模型权重获取

训练与推理需先下载权重到本地。下表为当前使用的 **PyTorch（Safetensors）** 与 **GGUF** 分发地址（约 2B 级、BF16，许可以各页面为准）。

| 用途 | 链接 |
|------|------|
| Transformers / 全量权重 | [Hugging Face 模型仓库](https://huggingface.co/fnholding/FN-LLM/tree/main) |
| GGUF / Ollama 等 | [Hugging Face GGUF 集合](https://huggingface.co/fnholding/FN-LLM-GGUF) |

推荐目录：

```text
models/
  base/                 # 放置 config.json、tokenizer、*.safetensors 等
```

GGUF 文件可置于 `ollama/`，文件名与 `ollama/Modelfile` 中 `FROM` 一致即可。

---

## 环境

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt   # CPT / SFT（LoRA）
```

需安装与 CUDA 匹配的 **PyTorch**；全量权重体积较大，请预留磁盘空间。

---

## 推理（Transformers）

默认加载 `./models/base`，或通过环境变量覆盖：

```bash
# Windows PowerShell
$env:BASE_MODEL_PATH="D:\path\to\base"
python scripts/infer.py
```

---

## Ollama（GGUF）

1. 从 [GGUF 集合](https://huggingface.co/fnholding/FN-LLM-GGUF) 下载量化文件到 `ollama/`，命名为 `base.gguf`（或修改 `ollama/Modelfile` 的 `FROM`）。
2. 在 `ollama` 目录执行：

```bash
cd ollama
ollama create base-model -f Modelfile
ollama run base-model:latest
```

`num_gpu` 等参数请按本机环境与 Ollama 版本调整。

---

## 训练流程（CPT → SFT）

1. **CPT**：在领域语料上做因果语言建模；注意学习率、与通用语料混合比例，减轻遗忘。训练入口指向 `models/base`（或你的 CPT 输出目录）。
2. **SFT（LoRA）**：在 CPT 产出或原始基座上，用指令数据监督微调；保持与 **chat template** 一致，可选用 PEFT、TRL、LLaMA-Factory 等流水线。

训练产出建议放在 `checkpoints/`（已加入 `.gitignore`）。

---

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── requirements-train.txt
├── .gitignore
├── models/
├── scripts/
│   └── infer.py
└── ollama/
    └── Modelfile
```

---

## 合规说明

若使用第三方公开发布的权重或衍生checkpoint，请遵守对应模型页的许可与引用要求。

---

## English

**Base Model** — project layout for agricultural-domain **CPT**, **LoRA SFT**, and local inference. **Weights are not included.** Place full checkpoints under `models/base`. **PyTorch weights:** [link](https://huggingface.co/fnholding/FN-LLM/tree/main). **GGUF:** [link](https://huggingface.co/fnholding/FN-LLM-GGUF). Run `python scripts/infer.py` after installing dependencies; set `BASE_MODEL_PATH` if needed.
