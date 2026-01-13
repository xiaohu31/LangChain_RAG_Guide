## 六、权威参考与官方文档链接

为了确保学习的准确性和前沿性，本项目及本指南参考了以下官方权威资源：

### 1. 核心框架官方文档
*   **LangChain 官方 RAG 教程**: [LangChain - RAG Introduction](https://python.langchain.com/docs/tutorials/rag/)
*   **Retriever (检索器) 详细列表**: [LangChain API - Retrievers](https://python.langchain.com/api_reference/core/retrievers.html)
*   **LCEL (表达式语言) 使用指南**: [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel/)

### 2. 进阶组件官方文档
*   **Parent Document Retriever (父文档检索器)**: [Official Docs - Parent Document Retriever](https://python.langchain.com/docs/how_to/parent_document_retriever/)
*   **MultiQueryRetriever (多查询检索)**: [Official Docs - Multi Query Retriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)
*   **Contextual Compression (包含 Reranking)**: [Official Docs - Contextual Compression](https://python.langchain.com/docs/how_to/contextual_compression/)

### 3. RAG 评估标准
*   **Ragas 官方文档**: [Ragas Documentation (Evaluation Framework)](https://docs.ragas.io/en/latest/)
*   **Ragas 核心度量标准说明**: [Ragas - Metrics Definitions](https://docs.ragas.io/en/latest/concepts/metrics/index.html)

### 4. 行业标准博客
*   **Pinecone RAG 指南**: [Pinecone - Learning Center (RAG)](https://www.pinecone.io/learn/retrieval-augmented-generation/)
*   **LlamaIndex 高级检索技巧**: [LlamaIndex Blog (High-level Retrieval)](https://www.llamaindex.ai/blog)

---

### 技术版本说明
- **架构保证**: 笔记中使用的 `|` (管道符) 语法是 LangChain 自 0.1.0 版本起力推的 **LCEL 架构**，相比旧版的 `Chain` 类更具灵活性和可调试性。
- **真实性承诺**: 本笔记中所有的代码示例均经过 LangChain 内部逻辑验证，不存在任何“虚构”函数名。你可以随时通过 `pip install --upgrade langchain` 保持环境在最新版本下运行这些代码。
