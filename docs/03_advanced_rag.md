

### 1. 带记忆的对话链 (Conversational RAG)

如果需要多轮对话（助手记住之前的聊天内容），需要引入历史记录处理。

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 第一步：创建历史感知检索器 ---
# 作用：处理用户问题中的代词（如“它”、“这个”），将其结合历史记录重写为一个独立完整的查询。

contextualize_q_system_prompt = """给定聊天历史记录和最新的用户问题（可能引用了聊天历史中的上下文），
请构造一个独立的问题，使其在没有聊天历史的情况下也能被理解。
不要回答问题，只需重写它，如果不需要重写则原样返回。"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"), # 聊天历史占位符
    ("human", "{input}"),                # 用户最新问题
])

# create_history_aware_retriever 会使用 LLM 来重写查询，然后使用检索器进行检索
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# --- 第二步：创建问答链 ---
# 作用：根据检索到的文档回答问题

qa_system_prompt = """你是一个问答助手。使用以下检索到的上下文来回答问题。
如果你不知道答案，就说不知道。保持回答简洁。

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# create_stuff_documents_chain 是最基本的文档处理链，它将所有文档拼接在一起放入 Prompt
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# --- 第三步：创建最终的 RAG 链 ---
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 使用示例 ---
from langchain_core.messages import HumanMessage, AIMessage

chat_history = [] # 初始化聊天记录

# 第一轮
response1 = rag_chain.invoke({"input": "LangChain 是什么?", "chat_history": chat_history})
print(response1["answer"])

# 更新历史
chat_history.extend([HumanMessage(content="LangChain 是什么?"), AIMessage(content=response1["answer"])])

# 第二轮 (指代 "它")
response2 = rag_chain.invoke({"input": "它支持 Python 吗?", "chat_history": chat_history})
print(response2["answer"])
```

### 2. 高级检索策略 (Advanced Retrieval)

为了提高检索准确率，可以使用 **混合检索 (Hybrid Search)**。

```python
# EnsembleRetriever 仍在 langchain.retrievers 中
# BM25Retriever 需要从 langchain_community 导入
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 1. BM25 检索器: 基于关键词匹配 (TF-IDF 的改进版)
# 优势: 对精确匹配、专有名词、特定错误代码等效果极佳
# 劣势: 无法理解语义 (如 "开心" 和 "高兴")
bm25_retriever = BM25Retriever.from_documents(split_docs)

# 2. 向量检索器: 基于语义匹配
# 优势: 理解语义关系
# 劣势: 对精确关键词可能不如 BM25
vector_retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# 3. 混合检索器 (Ensemble)
# 作用: 结合两者的优点，通过加权平均得出最终结果
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5] # 权重设置，可根据实际情况调整
)

# 在 RAG 链中使用混合检索器替代普通 retriever
rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### 3. 重排序 (Reranking) - 提升精度的“最后一步”

**知识点说明**: 向量检索（相似度搜索）只根据语义距离找 Top-k，但它并不真正“理解”问题。重排序是使用一个更强大的模型（Cross-Encoder）对初筛出的文档进行打分，确保最相关的文档排在第一位。

```python
# ContextualCompressionRetriever 在 langchain.retrievers 中
# FlashrankRerank 是第三方集成，在 langchain_community 中
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

# 1. 基础检索器
base_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# 2. 配置重排序器 (以 Flashrank 为例，轻量且快速)
# 作用: 将初步筛选的 10 个文档重新排列，只留下最有用的 3 个
compressor = FlashrankRerank(model="ms-marco-Minilm-L-6-v2", top_n=3)

# 3. 创建压缩检索器
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)

# 使用方式同普通 retriever
```

### 4. 父文档检索器 (Parent Document Retriever) - 解决“颗粒度”矛盾

**知识点说明**: 这是一个非常实用的技巧。
- **矛盾点**: 小块 (Small Chunks) 更有利于精准匹配向量，但大模型回答问题需要完整的上下文背景。
- **解决方案**: 将长文档切分为“父块”和“各级子块”。向量库里存子块，检索时匹配到子块，但返回给大模型的是它所属的“父块”内容。

```python
# LangChain v1.0+: ParentDocumentRetriever 可能需要从 langchain_classic 导入
# 如使用 v1.0+: from langchain_classic.retrievers import ParentDocumentRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# 定义父块和子块的分割器 (注意：需要 from langchain_text_splitters import RecursiveCharacterTextSplitter)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000) # 父块大
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)   # 子块小

vectorstore = Chroma(collection_name="split_parents", embedding_function=embedding_model)
store = InMemoryStore() # 存储完整的父文档内容

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加文档时，它会自动完成两级分割和关联
retriever.add_documents(pdf_docs, ids=None)
```

### 5. 多查询检索 (Multi-Query Retrieval) - 处理“提问不当”

**知识点说明**: 用户的问题往往比较简短或模糊。多查询法利用 LLM 将用户的一个问题改写成 3-5 个不同角度的提问，分别去库里搜，最后把结果去重汇总。

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
# LangChain v1.0+: 需从 langchain_classic.retrievers.multi_query 导入

# 只需要指定 LLM 和 基础检索器
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), 
    llm=llm
)

# 它会自动生成类似“请从技术角度描述...” “简述...”等多个变体
# 注意：get_relevant_documents() 已废弃，v1.0 起请使用 invoke()
unique_docs = multi_query_retriever.invoke("RAG原理")
```

### 6. RAG 效果评估 (Evaluation) - 拒绝盲目调优

**知识点说明**: 搭建完 RAG 后，如何量化它的表现？业界通用的评估框架是 **Ragas**。它关注四个维度（Ragas Metrics）：
1. **忠实度 (Faithfulness)**: 答案是否完全来自于检索到的内容？（防止幻觉）
2. **相关性 (Answer Relevance)**: 答案是否真的回答了用户的问题？
3. **上下文精度 (Context Precision)**: 检索到的文档里，真正有用的信息是否排在前面？
4. **上下文召回率 (Context Recall)**: 检索到的内容是否包含了回答问题的全部关键信息？

> **建议工具**: `ragas` 库。通过 LLM-as-a-Judge（让更强的模型如 GPT-4 来给当前模型的回答打分）来实现自动化评估。

```python
# Ragas 评估示例
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# 准备评估数据集
eval_data = {
    "question": ["什么是 RAG？"],
    "answer": ["RAG 是检索增强生成技术..."],
    "contexts": [["RAG (Retrieval-Augmented Generation) 的核心思想是..."]],
    "ground_truth": ["RAG 是一种结合检索和生成的技术"]
}
eval_dataset = Dataset.from_dict(eval_data)

# 执行评估
result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(result)  # 输出各维度评分
```

### 7. 索引 API (Indexing API) - 生产环境必备
**核心价值**: 当你的本地文件发生变动（新增、修改、删除）时，如果你重新跑一次 `from_documents`，会造成大量重复的向量存储和昂贵的 API 费用。Indexing API 会对比文件指纹，**仅同步变动部分**。

```python
# LangChain v1.0+: Indexing API 已移至 langchain-classic
# 如使用 v1.0+: from langchain_classic.indexes import index
from langchain.indexes import index

# 1. 定义 Record Manager (记录管理器)，通常存在本地数据库
from langchain_community.indexes import SQLRecordManager
record_manager = SQLRecordManager("sqlite:///record_manager_cache.sql", namespace="my_rag_app")
record_manager.create_schema()

# 2. 执行索引动作 (cleanup="incremental" 代表增量同步)
indexing_stats = index(
    split_docs,
    record_manager,
    vectordb,
    cleanup="incremental",
    source_id_key="source"
)
# 返回值包含：num_added, num_updated, num_deleted, num_skipped
```

### 8. 结构化输出 (Structured Output) - 现代 RAG 的标配
**核心价值**: 在 v0.3 中，官方推荐直接将 LLM 输出绑定到 Pydantic 模型，确保下游系统可以稳定解析结果。

```python
from pydantic import BaseModel, Field

# 定义期望得到的回答结构
class AnswerSchema(BaseModel):
    answer: str = Field(description="对问题的最终回答")
    sources: list[str] = Field(description="回答时引用的具体源文件路径")
    ref_score: float = Field(description="该回答与上下文的相关度评分(0-1)")

# 绑定结构化输出
structured_llm = llm.with_structured_output(AnswerSchema)

# 在 RAG 链中使用
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | structured_llm
)
```

### 9. 代理型 RAG 与 create_agent — LangChain v1.0 新标准

**趋势说明**: 2025 年的趋势是由“链”转向“图”。Agent 不再仅仅是简单的检索，它会判断：
1. **Query Analysis**: 这个问题需要搜库吗？（比如问“你好”，Agent 会直接回，不搜库）
2. **Self-Correction**: 如果搜出来的东西没用，Agent 会自动重写问题再搜一次。
3. **Tool Choice**: 这个问题是在知识库里，还是需要联网去查？

> **参考资源**: 建议学习官方的 [LangGraph 框架](https://python.langchain.com/docs/concepts/langgraph/)，它是实现这种自适应、循环式 RAG 的新标准。

**create_agent 基本用法** (LangChain v1.0+):

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中检索相关信息"""
    docs = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)

# 创建 Agent (v1.0 新标准 API)
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[search_knowledge_base],
)

# 执行 Agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "什么是 RAG 技术？"}]
})
print(result["messages"][-1].content)
```

**中间件 (Middleware)** — v1.0 核心新特性:

```python
# 中间件允许在 Agent 执行循环中注入自定义逻辑
def logging_middleware(state, config, next_step):
    """日志中间件：记录每次调用"""
    print(f"[LOG] 处理 {len(state['messages'])} 条消息")
    result = next_step(state, config)
    print(f"[LOG] 完成处理")
    return result

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[search_knowledge_base],
    middleware=[logging_middleware]  # 注入中间件
)
```

### 10. 查询分析与语义路由 (Query Analysis & Routing)

**核心价值**: 并非所有问题都需要查同一个库。路由层可以根据用户意图，将问题分发给最合适的检索器（如：技术手册库 vs 销售数据SQL库 vs 闲聊）。

```python
from langchain.utils.math import cosine_similarity
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

# 1. 定义路由模板
physics_template = """你是一个物理学教授。请回答以下物理问题: {query}"""
math_template = """你是一个数学家。请回答以下数学问题: {query}"""

routes = {"physics": physics_template, "math": math_template}
embeddings = OpenAIEmbeddings()
route_embeddings = embeddings.embed_documents(list(routes.values()))

def route(info):
    query_embedding = embeddings.embed_query(info["query"])
    similarity = cosine_similarity([query_embedding], route_embeddings)[0]
    most_similar = list(routes.keys())[similarity.argmax()]
    return routes[most_similar]

# 动态路由链
chain = ({"query": RunnablePassthrough()} | RunnableLambda(route) | llm)
```

### 11. 多模态 RAG (Multimodal RAG) - 2026 年前沿趋势

**趋势说明**: 未来的 RAG 不仅仅是搜文字。多模态 RAG 允许你：
1. **图文检索**: 用户问"产品外观是什么样的？"，系统可以检索并返回产品图片。
2. **视频理解**: 从视频中提取关键帧并进行语义检索。
3. **统一向量空间**: 使用如 CLIP、Jina CLIP 等模型将文本和图像嵌入到同一个向量空间。

```python
# 示例：使用 Jina CLIP 进行多模态嵌入
from langchain_community.embeddings import JinaEmbeddings

embeddings = JinaEmbeddings(
    jina_api_key="YOUR_API_KEY",
    model_name="jina-clip-v2"
)
# 文字和图片可以放在同一个向量库里进行混合检索
```

> **参考资源**: [LangChain - Multimodal](https://python.langchain.com/docs/how_to/#multimodal)

### 12. LangSmith 可观测性 — 生产环境必备

**核心价值**: 生产级 RAG 应用必须具备可观测性。LangSmith 是 LangChain 官方推荐的追踪、评估、调试工具。

**主要功能**:
1. **Trace 追踪**: 可视化查看每次请求的完整调用链
2. **评估测试**: 自动化测试 RAG 输出质量
3. **Prompt 版本管理**: 管理和迭代 Prompt 模板
4. **性能监控**: 监控延迟、Token 消耗等指标

```python
# 1. 设置环境变量启用 LangSmith
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"  # 从 smith.langchain.com 获取
os.environ["LANGCHAIN_PROJECT"] = "my-rag-project"

# 2. 正常使用 LangChain，追踪自动生效
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("回答问题: {question}")
chain = prompt | llm

# 每次调用都会自动记录到 LangSmith
result = chain.invoke({"question": "什么是 RAG？"})
# 在 smith.langchain.com 查看完整追踪信息
```

> **参考资源**: [LangSmith 官方文档](https://docs.smith.langchain.com/)

---
