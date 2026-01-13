## 二、详细实现步骤

### 0. 环境准备与依赖安装

首先安装 LangChain 生态系统的核心库和常用组件。

```bash
# langchain: 核心框架 (v1.0+)
# langchain-community: 社区组件(包含各种 loaders, vectorstores)
# langchain-openai: OpenAI 模型封装
# langchain-classic: 【重要】传统链和检索器（如 LLMChain, MultiQueryRetriever 等）
# chromadb: 向量数据库
# pydantic: v1.0 全面升级至 Pydantic v2
pip install -U langchain langchain-community langchain-openai langchain-classic chromadb pydantic
```

> ⚠️ **版本兼容性警告 (2026年1月更新)**:
> LangChain 已于 **2025 年 10 月**正式发布 **v1.0**。许多传统功能（如 `ConversationalRetrievalChain`, `MultiQueryRetriever`, `RetrievalQA`, Indexing API 等）已从核心 `langchain` 包移至 **`langchain-classic`** 包。
> - **如果你使用 LangChain >= 1.0**: 需要将 `from langchain.chains import ...` 改为 `from langchain_classic.chains import ...`，将 `from langchain.retrievers import MultiQueryRetriever` 改为 `from langchain_classic.retrievers import MultiQueryRetriever`。
> - **如果你使用 LangChain 0.2.x / 0.3.x**: 本笔记中的旧导入路径可以直接使用。
> - **Text Splitter**: 已迁移至独立包 `langchain_text_splitters`。
> - 请参考官方迁移指南: [LangChain v1 Migration Guide](https://python.langchain.com/docs/versions/v1/)

### 1. 文档加载 (Document Loading)

LangChain 提供了多种 `Loader` 来处理不同格式的文件。

```python
from langchain_community.document_loaders import (
    PyMuPDFLoader,              # 专用于 PDF 文件，解析速度快，效果好
    UnstructuredMarkdownLoader, # 用于 Markdown 文件
    TextLoader,                 # 用于纯文本文件 (.txt)
    WebBaseLoader               # 用于爬取和解析网页内容
)

# 1.1 加载 PDF 文件
# PyMuPDFLoader 会将 PDF 的每一页加载为一个 Document 对象
# Document 对象包含 page_content (文本内容) 和 metadata (元数据，如页码、文件名)
pdf_loader = PyMuPDFLoader("./data/knowledge.pdf")
pdf_docs = pdf_loader.load()

# 1.2 加载 Markdown 文件
# UnstructuredMarkdownLoader 会解析 Markdown 结构
md_loader = UnstructuredMarkdownLoader("./data/readme.md")
md_docs = md_loader.load()

# 1.3 加载网页
# bs_kwargs 用于指定 BeautifulSoup 的解析参数，这里只提取 article 标签的内容
from bs4 import SoupStrainer
web_loader = WebBaseLoader(
    web_paths=("https://example.com/article",),
    bs_kwargs=dict(parse_only=SoupStrainer("article")) 
)
web_docs = web_loader.load()

print(f"加载了 {len(pdf_docs)} 页 PDF 文档")
```

### 2. 文本分割 (Text Splitting)

将长文档切分为较小的块 (Chunks)，以便于 Embedding 和适应 LLM 的上下文窗口。

```python
# 注意：LangChain v0.2+ 起，text_splitter 已迁移至独立包
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 初始化分割器
# RecursiveCharacterTextSplitter 是最常用的分割器，它会递归地尝试按分隔符列表进行分割，
# 优先保持段落、句子的完整性。
text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size: 每个分块的最大字符数。建议值: 500-1000
    # 太小会导致语义破碎，太大会导致检索不精准
    chunk_size=500,
    
    # chunk_overlap: 分块之间的重叠字符数。
    # 作用: 保持上下文连贯性，避免句子被切断导致语义丢失。建议值: chunk_size 的 10%-20%
    chunk_overlap=50,
    
    # length_function: 用于计算长度的函数，默认是 len() 计算字符数
    length_function=len,
    
    # separators: 分隔符列表，按优先级从左到右尝试分割
    separators=["\n\n", "\n", "。", "！", "？", ";", "；", " ", ""]
)

# 执行分割
# docs 是上一步加载的文档列表
# split_documents 方法会返回一个新的 Document 列表，包含分割后的文本块
split_docs = text_splitter.split_documents(pdf_docs)

print(f"分割后共有 {len(split_docs)} 个文本块")
```

### 3. 文本向量化 (Embeddings)

选择一个 Embedding 模型将文本转换为向量。

```python
# 选项 A: 使用 OpenAI Embeddings (需要 API Key，效果好，收费)
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small", # OpenAI 的新一代高效模型
    openai_api_key="sk-..."         # 你的 OpenAI API Key
)

# 选项 B: 使用本地 HuggingFace 模型 (免费，隐私好，需要计算资源)
# 2026 年推荐模型:
#   - 'BAAI/bge-m3' (多语言 + 多粒度，2025年新模型)
#   - 'jinaai/jina-embeddings-v3' (全球排名前列)
#   - 'moka-ai/m3e-base' (中文经典模型)
from langchain_huggingface import HuggingFaceEmbeddings

# model_kwargs={'device': 'cpu'} 指定运行设备，有 GPU 可改为 'cuda'
embedding_model = HuggingFaceEmbeddings(
    model_name="moka-ai/m3e-base",
    model_kwargs={'device': 'cpu'} 
)
```

### 4. 向量数据库存储 (Vector Stores)

将切分好的文本块和对应的向量存储到向量数据库中。

```python
from langchain_community.vectorstores import Chroma

# 定义持久化存储路径，这样重启程序后数据不会丢失
persist_directory = "./vector_db_data"

# 创建并保存向量库
# from_documents 方法会执行以下操作：
# 1. 调用 embedding_model 将 split_docs 中的文本转换为向量
# 2. 将向量和原始文本存储到 Chroma 数据库中
# 3. 将数据持久化到 persist_directory
vectordb = Chroma.from_documents(
    documents=split_docs,           # 分割后的文档列表
    embedding=embedding_model,      # 使用的 Embedding 模型
    persist_directory=persist_directory # 持久化目录
)

# 如果需要加载已存在的向量库，使用以下代码:
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

print("向量库创建完成并已持久化")
```

### 5. 检索与 LLM 生成 (RAG Chain)

这是 RAG 的核心部分：检索 -> 增强 -> 生成。

#### 5.1 配置 LLM

```python
from langchain_openai import ChatOpenAI

# 初始化大语言模型
# model_name: 指定使用的模型版本
# temperature: 控制输出的随机性。0 表示最确定、最事实；1 表示最有创意。
# RAG 任务通常建议设为 0，以防止模型产生幻觉。
llm = ChatOpenAI(
    model_name="gpt-4o-mini",  # 2026年性价比最高的模型 (或使用 gpt-4o 获得最佳效果)
    temperature=0,              
    openai_api_key="sk-..."
)
```

#### 5.2 构建 Prompt 模板

```python
from langchain_core.prompts import ChatPromptTemplate

# 定义 Prompt 模板
# {context}: 占位符，将被替换为检索到的文档片段
# {question}: 占位符，将被替换为用户的问题
template = """你是一个专业的知识库助手。请根据以下提供的上下文信息回答用户的问题。

规则:
1. 如果上下文信息不足以回答问题，请直接说"我根据已知信息无法回答该问题"，不要编造。
2. 回答要简洁明了。

上下文信息:
{context}

用户问题: {question}

回答:"""

prompt = ChatPromptTemplate.from_template(template)
```

#### 5.3 构建 LCEL 链 (LangChain Expression Language)

LCEL 是 LangChain 推荐的构建方式，它使用 Linux 管道风格的语法 (`|`) 将组件连接起来。

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 将向量库转换为检索器 (Retriever)
# search_type="similarity": 使用余弦相似度搜索
# k=4: 每次检索返回最相似的 4 个文档块
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def format_docs(docs):
    """辅助函数: 将检索到的 Document 对象列表转换为纯字符串，用换行符连接"""
    return "\n\n".join(doc.page_content for doc in docs)

# 构建 RAG 流水线
# 字典中的 key (context, question) 对应 Prompt 模板中的变量名
rag_chain = (
    {
        "context": retriever | format_docs,  # 步骤 1: 调用检索器获取文档，并格式化为字符串
        "question": RunnablePassthrough()    # 步骤 2: 传递用户原始问题
    }
    | prompt                                 # 步骤 3: 将 context 和 question 填充到 Prompt 模板
    | llm                                    # 步骤 4: 将完整的 Prompt 发送给 LLM
    | StrOutputParser()                      # 步骤 5: 将 LLM 的输出对象解析为纯文本字符串
)

# 执行查询
query = "什么是 RAG 技术？"
response = rag_chain.invoke(query)
print(f"问题: {query}")
print(f"回答: {response}")
```

---
