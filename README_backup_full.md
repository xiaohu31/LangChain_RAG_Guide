# LangChain RAG (检索增强生成) 实现完整指南

本指南详细介绍了如何使用 LangChain 框架从零构建一个 RAG 系统。涵盖了从文档加载、切分、向量化、存储到检索和生成的全流程，并附带了详细的代码注释。

---

## 📚 目录

- [一、RAG 核心流程](#一rag-核心流程)
- [二、详细实现步骤](#二详细实现步骤)
  - [0. 环境准备与依赖安装](#0-环境准备与依赖安装)
  - [1. 文档加载](#1-文档加载-document-loading)
  - [2. 文本分割](#2-文本分割-text-splitting)
  - [3. 文本向量化](#3-文本向量化-embeddings)
  - [4. 向量数据库存储](#4-向量数据库存储-vector-stores)
  - [5. 检索与 LLM 生成](#5-检索与-llm-生成-rag-chain)
- [三、高级功能与进阶](#三高级功能与进阶)
  - [1. 带记忆的对话链](#1-带记忆的对话链-conversational-rag)
  - [2. 高级检索策略](#2-高级检索策略-advanced-retrieval)
  - [3. 重排序 (Reranking)](#3-重排序-reranking---提升精度的最后一步)
  - [4. 父文档检索器](#4-父文档检索器-parent-document-retriever---解决颗粒度矛盾)
  - [5. 多查询检索](#5-多查询检索-multi-query-retrieval---处理提问不当)
  - [6. RAG 效果评估](#6-rag-效果评估-evaluation---拒绝盲目调优)
    - [6.1 使用 Ragas 库评估](#61-使用-ragas-库评估开源方案)
    - [6.2 使用 LangSmith 评估](#62-使用-langsmith-评估官方推荐)
  - [7. 索引 API](#7-索引-api-indexing-api---生产环境必备)
  - [8. 结构化输出](#8-结构化输出-structured-output---现代-rag-的标配)
  - [9. 代理型 RAG 与 create_agent](#9-代理型-rag-与-create_agent--langchain-最新标准)
    - [9.1 使用 @tool 创建检索工具](#91-使用-tool-创建检索工具)
    - [9.2 创建 RAG Agent](#92-创建-rag-agent官方标准用法)
    - [9.3 使用 dynamic_prompt 中间件](#93-使用-dynamic_prompt-中间件实现-rag-链)
    - [9.4 返回源文档](#94-返回源文档带上下文状态管理)
    - [9.5 LangGraph Agentic RAG 完整工作流](#95-langgraph-agentic-rag-完整工作流)
  - [10. 查询分析与语义路由](#10-查询分析与语义路由-query-analysis--routing)
  - [11. 多模态 RAG](#11-多模态-rag-multimodal-rag---2026-年前沿趋势)
  - [12. LangSmith 可观测性](#12-langsmith-可观测性---生产环境必备)
- [四、2026 年 RAG 前沿技术](#四2026-年-rag-前沿技术-cutting-edge)
- [五、RAG 性能调优 Checklist](#五总结rag-性能调优-checklist-2026-更新版)
- [六、权威参考与官方文档链接](#六权威参考与官方文档链接)

---

## 🚀 快速入门 (30秒上手)

以下是一个最简 RAG 示例，帮助你快速理解核心流程：

```python
# 安装依赖: pip install langchain langchain-openai langchain-community chromadb langchain_text_splitters

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 加载文档
docs = TextLoader("./your_document.txt").load()

# 2. 分割文档
splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

# 3. 创建向量库
vectordb = Chroma.from_documents(splits, OpenAIEmbeddings())

# 4. 构建 RAG 链
prompt = ChatPromptTemplate.from_template("根据以下内容回答问题:\n{context}\n\n问题: {question}")
rag_chain = (
    {"context": vectordb.as_retriever() | (lambda docs: "\n".join(d.page_content for d in docs)), 
     "question": RunnablePassthrough()}
    | prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
)

# 5. 提问
print(rag_chain.invoke("文档的主要内容是什么？"))
```

---

## 📚 目录 (Table of Contents)

- [一、RAG 核心流程](docs/01_core_principles.md)
- [二、详细实现步骤](docs/02_step_by_step_guide.md)
- [三、高级功能与进阶](docs/03_advanced_rag.md)
- [四、2026 年 RAG 前沿技术](docs/04_cutting_edge_2026.md)
- [五、RAG 性能调优 Checklist](docs/05_performance_checklist.md)
- [六、权威参考与官方文档链接](docs/06_references.md)

---

> 💡 本文档由自动化脚本拆分生成，详细内容请点击上方链接查看各章节。