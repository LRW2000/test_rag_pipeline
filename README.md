# test_rag_pipeline

### 使用说明

1. **准备环境**：
   ```bash
   conda create --name rag_pipeline python=3.10
   
   conda activate rag_pipeline
   
   pip install langchain faiss-cpu sentence-transformers transformers rouge-score nltk jieba bert-score -i https://pypi.tuna.tsinghua.edu.cn/simple/

   pip install -U langchain-community -i https://pypi.tuna.tsinghua.edu.cn/simple

   pip install tiktoken protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple

   pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple

   ```

2. **文件结构准备**：
   ```
   ├── docs/                 # 存放原始文档
   ├── models/               # 存放本地模型
   │   ├── embedding_model/  # Embedding模型
   │   ├── reranker/         # Reranker模型
   │   └── llm/              # 大语言模型
   ├── test_data.json        # 测试集文件
   └── rag_pipeline.py       # 本脚本文件

   模型路径可以自己设定
   ```
 

3. **测试集格式示例**（test_data.json）：
   ```json
   [
       {
           "question": "量子计算的主要特点是什么？",
           "relevant_chunks": ["量子计算利用量子比特...", "量子并行计算可以..."],
           "answer": "量子计算的主要特点是..."
       }
   ]
   ```

4. **运行脚本**：
   ```bash
   python rag_pipeline.py
   ```

### 关键功能说明

1. **模块化设计**：每个处理阶段都有明确分离，便于单独调试和扩展
2. **本地模型支持**：所有模型均从本地路径加载，支持HuggingFace格式的模型
3. **评估指标**：
   - ROUGE-1/ROUGE-L：衡量生成文本与参考文本的重叠度
   - BLEU：衡量n-gram匹配精度
   - BERTScore：基于上下文的语义相似度评估
4. **中间结果保存**：所有中间处理结果均保存为JSON文件，方便问题排查和过程分析
