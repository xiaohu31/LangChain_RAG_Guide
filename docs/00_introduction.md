# LangChain RAG (检索增强生成) 实现完整指南

本指南详细介绍了如何使用 LangChain 框架从零构建一个 RAG 系统。涵盖了从文档加载、切分、向量化、存储到检索和生成的全流程，并附带了详细的代码注释。

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
