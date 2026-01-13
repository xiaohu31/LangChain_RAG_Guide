

RAG (Retrieval-Augmented Generation) 的核心思想是：先检索相关信息，再辅助大模型生成答案。

**流程图解:**
1. **Load (加载)**: 将 PDF、Word、Markdown 等文件加载为文本。
2. **Split (分割)**: 将长文本分割为较小的块 (Chunks)。
3. **Embed (向量化)**: 将文本块转换为数值向量。
4. **Store (存储)**: 将向量存储到向量数据库 (Vector DB)。
5. **Retrieve (检索)**: 根据用户问题，在向量库中查找最相似的文本块。
6. **Generate (生成)**: 将检索到的文本块作为“上下文”喂给 LLM，生成最终答案。

---
