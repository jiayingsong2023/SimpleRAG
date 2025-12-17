# Simple RAG with DeepSeek & ReRank

这是一个基于 Python 的 RAG (Retrieval-Augmented Generation) 应用程序，使用 **DeepSeek** 大模型，结合 **ReRank (重排序)** 技术，实现了基于本地 PDF 文档的高精度智能问答系统。

## ✨ 功能特点

*   **本地文档处理**: 自动扫描并加载指定目录下的 PDF 文件。
*   **双阶段检索**:
    1.  **Recall (召回)**: 使用 `sentence-transformers/all-MiniLM-L6-v2` 向量模型进行初步检索 (Top-20)。
    2.  **ReRank (重排序)**: 使用 `BAAI/bge-reranker-base` Cross-Encoder 模型对候选文档进行精细打分 (Top-5)。
*   **高性能向量库**: 使用 ChromaDB 存储向量数据，支持持久化保存。
*   **DeepSeek LLM**: 集成 DeepSeek V3 (via OpenAI API) 提供强大的推理和回答能力。
*   **智能检索**:
    *   自动检测数据库状态，为空时自动重建索引。
    *   支持显示检索到的参考来源及**相关性得分**。
    *   调试模式：可查看发送给大模型的完整 Prompt 和上下文。

## 🛠️ 技术栈

*   [LangChain](https://www.langchain.com/) - LLM 应用开发框架
*   [DeepSeek API](https://platform.deepseek.com/) - 大语言模型
*   [ChromaDB](https://www.trychroma.com/) - 向量数据库
*   [Sentence Transformers](https://www.sbert.net/) - 向量化与重排序 (CrossEncoder)
*   [uv](https://github.com/astral-sh/uv) - 极速 Python 包管理器

## 🚀 快速开始

### 1. 环境准备

确保已安装 Python 3.10+。本项目推荐使用 `uv` 进行依赖管理。

### 2. 安装依赖

```bash
# 如果使用 uv (推荐)
uv sync

# 或者使用 pip
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件，并填入你的 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. 准备文档

将你的 PDF 文档放入以下目录（可在代码中修改 `DOCS_PATH`）：
`C:\Users\Administrator\Documents\EBOOK`

### 5. 运行程序

```bash
# 使用 uv 运行
uv run python rag_app.py

# 或者直接运行
python rag_app.py
```

## 💡 使用说明

1.  程序启动时会自动加载本地 Embedding 模型和 ReRank 模型（首次运行会自动下载模型）。
2.  如果是首次运行或数据库为空，程序会自动扫描 PDF 目录并构建向量索引。
3.  系统就绪后，在提示符下输入你的问题。
4.  程序会执行 **检索 -> 重排序 -> 生成** 流程，并返回 AI 的回答及参考文档（含相关性得分）。
5.  输入 `exit` 或 `quit` 退出程序。

## ⚠️ 常见问题

*   **Insufficient Balance**: 请检查 DeepSeek 账户余额。
*   **检索结果为空**: 可能是 PDF 目录为空，或者数据库需要重建。尝试删除 `chroma_db` 文件夹后重启程序。
*   **模型下载慢**: 首次运行需要下载 Embedding 和 ReRank 模型，请耐心等待。
