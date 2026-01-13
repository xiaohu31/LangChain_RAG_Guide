import re
import os
import subprocess

def split_readme():
    output_dir = 'docs'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read content from git history (previous commit) to recover full content
    try:
        content = subprocess.check_output(['git', 'show', 'HEAD^:README.md']).decode('utf-8')
        print("Successfully read content from git history.")
    except Exception as e:
        print(f"Error reading from git: {e}")
        return

    # Normalizing content headers just in case
    # The observed headers are "## 一、", "## 二、" etc.
    
    headers = [
        ("一、RAG 核心流程", "01_core_principles.md"),
        ("二、详细实现步骤", "02_step_by_step_guide.md"),
        ("三、高级功能与进阶", "03_advanced_rag.md"),
        ("四、2026 年 RAG 前沿技术", "04_cutting_edge_2026.md"),
        ("五、总结：RAG 性能调优 Checklist (2026 更新版)", "05_performance_checklist.md"),
        ("六、权威参考与官方文档链接", "06_references.md")
    ]
    
    intro_filename = "00_introduction.md"
    indices = []
    
    for header, filename in headers:
        # Use simple string search first as it is more robust if format is exact
        pattern = f"## {header}"
        pos = content.find(pattern)
        if pos != -1:
            indices.append((pos, header, filename))
        else:
            print(f"Warning: Header '{header}' not found!")

    indices.sort()
    
    if indices:
        intro_content = content[:indices[0][0]].strip()
        # Remove TOC if present
        intro_content = re.sub(r'## 📚 目录.*?(?=## 🚀)', '', intro_content, flags=re.DOTALL).strip()
        
        with open(os.path.join(output_dir, intro_filename), 'w', encoding='utf-8') as f:
            f.write(intro_content + "\n")
        print(f"Created {intro_filename}")

    for i in range(len(indices)):
        start_pos = indices[i][0]
        filename = indices[i][2]
        
        if i < len(indices) - 1:
            end_pos = indices[i+1][0]
            section_content = content[start_pos:end_pos].strip()
        else:
            section_content = content[start_pos:].strip()
            
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write(section_content + "\n")
        print(f"Created {filename}")

    # Create new Index README.md (keeps existing index structure we planned)
    new_readme_content = f"""# LangChain RAG 完整指南 (2026 版)

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

---

*上次更新: 2026-01-13*
"""

    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(new_readme_content)
    print("Updated README.md with new index.")

if __name__ == "__main__":
    split_readme()

