# LangChain RAG 完整指南 (2026 版)

本指南详细介绍了如何使用 LangChain 框架从零构建一个 RAG 系统。为方便阅读与维护，已将文档拆分为多个章节。

## 📚 目录

- **[0. 快速入门](./docs/00_introduction.md)**
  - 核心概念简介
  - 30秒上手极简示例

- **[1. RAG 核心流程](./docs/01_core_principles.md)**
  - Load -> Split -> Embed -> Store -> Retrieve -> Generate 全流程图解

- **[2. 详细实现步骤](./docs/02_step_by_step_guide.md)**
  - 包含环境配置、文档加载、分割、向量化等完整代码实现

- **[3. 高级功能与进阶](./docs/03_advanced_rag.md)**
  - 记忆对话 (Conversational RAG)
  - 混合检索 (Hybrid Search)
  - 重排序 (Reranking)
  - 代理型 RAG (Agentic RAG) & LangGraph 工作流
  - RAG 评估 (Ragas & LangSmith)

- **[4. 2026 前沿技术](./docs/04_cutting_edge_2026.md)**
  - GraphRAG
  - Corrective RAG (CRAG)
  - 多模态 RAG
  - LangGraph 2.0 展望

- **[5. 性能调优 Checklist](./docs/05_performance_checklist.md)**
  - 生产环境排查清单

- **[6. 参考资源](./docs/06_references.md)**
  - 官方文档与权威博客链接

## 关于本项目 (About)

### 🎯 项目简介

**LangChain RAG 指南** 是一个旨在帮助开发者从零开始掌握 Retrieval-Augmented Generation (检索增强生成) 技术的开源项目。

本项目基于最新的 LangChain 框架（v1.0+），详细拆解了 RAG 系统的每一个关键环节：从文档处理、向量化存储，到高级检索策略和 LLM 生成。无论你是初学者还是有经验的开发者，都能在这里找到实用的代码示例和最佳实践。

### 💡 核心目标

- **系统化学习**: 提供一条清晰的 RAG 技术学习路径。
- **实战导向**: 所有代码均可运行，覆盖从简单 Demo 到生产级应用的多种场景。
- **紧跟前沿**: 包含 GraphRAG、多模态检索、Agentic RAG 等 2026 年最新趋势。

### 👥 适用人群

- 想要构建私有知识库问答系统的开发者
- 正在学习 LangChain 和大模型应用开发的 AI 爱好者
- 希望优化现有 RAG 系统性能的算法工程师

### 🛠️ 技术栈

- **LangChain**: 核心编排框架
- **OpenAI / HuggingFace**: 模型服务
- **ChromaDB**: 向量数据库
- **Docsify**: 文档网站构建工具

### 🔗 关于作者

本项目由 **[@xiaohu31](https://github.com/xiaohu31)** 维护。

如果您发现任何错误或有改进建议，欢迎提交 Issue 或 Pull Request！

- **GitHub 仓库**: [https://github.com/xiaohu31/LangChain_RAG_Guide](https://github.com/xiaohu31/LangChain_RAG_Guide)

### 📄 开源协议

MIT License

---

*上次更新: 2026-01-13*
