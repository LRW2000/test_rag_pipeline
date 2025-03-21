以下是一个使用LangChain和Python实现RAG系统的完整pipeline示例代码，包含各环节结果保存和评估指标计算。请根据实际路径调整模型路径和文件路径。

```python
import os
import json
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import jieba
from bert_score import score

# 配置参数
class Config:
    # 输入路径
    doc_dir = "./docs"               # 文档目录
    test_data_path = "./test_data.json"  # 测试集路径
    embedding_model_path = "./models/embedding_model"  # 本地embedding模型路径
    reranker_model_path = "./models/reranker"          # 本地reranker模型路径
    llm_path = "./models/llm"                          # 本地大模型路径

    # 输出路径
    output_dir = "./output"
    chunk_path = "./output/chunks.json"
    faiss_path = "./output/faiss_index"
    retrieval_path = "./output/retrieval_results.json"
    rerank_path = "./output/rerank_results.json"
    answer_path = "./output/generated_answers.json"
    eval_path = "./output/evaluation_results.json"

    # 参数设置
    chunk_size = 500       # 文本分块长度
    chunk_overlap = 50     # 分块重叠长度
    top_k = 5              # 首次检索数量
    rerank_top_k = 3       # 重排序后保留数量

config = Config()

# 创建输出目录
os.makedirs(config.output_dir, exist_ok=True)

# 1. 文档加载和分块
def load_and_chunk():
    print("Loading and chunking documents...")
    documents = []
    for file in os.listdir(config.doc_dir):
        file_path = os.path.join(config.doc_dir, file)
        try:
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                continue
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    # 保存分块结果
    chunk_data = [{"text": doc.page_content, "metadata": doc.metadata} for doc in chunks]
    with open(config.chunk_path, 'w') as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
    return chunks

# 2. 向量化并构建FAISS索引
def create_vector_store(chunks):
    print("Creating vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model_path,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(config.faiss_path)
    return vectorstore

# 3. 首次检索
def first_retrieval(test_data, vectorstore):
    print("Running first retrieval...")
    results = []
    for item in tqdm(test_data):
        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        # 相似度检索
        docs = vectorstore.similarity_search(question, k=config.top_k)
        retrieved = [doc.page_content for doc in docs]

        # 计算召回率
        hit = sum(1 for c in retrieved if c in relevant_chunks)
        recall = hit / len(relevant_chunks) if len(relevant_chunks) > 0 else 0

        results.append({
            "question": question,
            "retrieved_chunks": retrieved,
            "recall": recall,
            "relevant_chunks": relevant_chunks
        })

    with open(config.retrieval_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results

# 4. 重排序
def rerank_results(retrieval_results):
    print("Running reranking...")
    reranker = CrossEncoder(config.reranker_model_path)

    results = []
    for item in tqdm(retrieval_results):
        question = item["question"]
        chunks = item["retrieved_chunks"]
        relevant_chunks = item["relevant_chunks"]

        # 准备输入对
        pairs = [(question, chunk) for chunk in chunks]

        # 获取分数
        scores = reranker.predict(pairs)

        # 重新排序
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        reranked = [chunk for chunk, _ in ranked[:config.rerank_top_k]]

        # 计算重排召回率
        hit = sum(1 for c in reranked if c in relevant_chunks)
        recall_rerank = hit / len(relevant_chunks) if len(relevant_chunks) > 0 else 0

        results.append({
            "question": question,
            "reranked_chunks": reranked,
            "recall_rerank": recall_rerank
        })

    with open(config.rerank_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results

# 5. 生成答案
def generate_answers(reranked_results):
    print("Generating answers...")
    # 加载本地大模型
    tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.llm_path,
        device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    results = []
    for item in tqdm(reranked_results):
        context = "\n".join(item["reranked_chunks"])
        prompt = f"根据以下上下文回答问题：\n上下文：{context}\n问题：{item['question']}\n答案："

        try:
            answer = llm(prompt)
        except Exception as e:
            answer = f"生成错误：{str(e)}"

        results.append({
            "question": item["question"],
            "generated_answer": answer,
            "context": context
        })

    with open(config.answer_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results

# 6. 评估答案
def evaluate_answers(generated_answers, test_data):
    print("Evaluating answers...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    evaluations = []
    references = []
    candidates = []

    # 准备数据
    for gen, ref in zip(generated_answers, test_data):
        candidates.append(gen["generated_answer"])
        references.append(ref["answer"])

    # 计算BERTScore
    P, R, F1 = score(candidates, references, lang="zh")

    # 逐个样本计算指标
    for i, (cand, ref) in enumerate(zip(candidates, references)):
        # ROUGE
        rouge = scorer.score(ref, cand)

        # BLEU
        ref_tokens = list(jieba.cut(ref))
        cand_tokens = list(jieba.cut(cand))
        bleu_score = sentence_bleu([ref_tokens], cand_tokens)

        evaluations.append({
            "rouge1": rouge["rouge1"].fmeasure,
            "rougeL": rouge["rougeL"].fmeasure,
            "bleu": bleu_score,
            "bert_precision": P[i].item(),
            "bert_recall": R[i].item(),
            "bert_f1": F1[i].item()
        })

    with open(config.eval_path, 'w') as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)
    return evaluations

# 主流程
def main():
    # 1. 加载和分块
    chunks = load_and_chunk()

    # 2. 创建向量库
    vectorstore = create_vector_store(chunks)

    # 加载测试数据
    with open(config.test_data_path) as f:
        test_data = json.load(f)

    # 3. 首次检索
    retrieval_results = first_retrieval(test_data, vectorstore)

    # 4. 重排序
    reranked_results = rerank_results(retrieval_results)

    # 5. 生成答案
    generated_answers = generate_answers(reranked_results)

    # 6. 评估
    evaluation_results = evaluate_answers(generated_answers, test_data)

    # 打印平均指标
    avg_metrics = {
        "rouge1": sum(e["rouge1"] for e in evaluation_results) / len(evaluation_results),
        "rougeL": sum(e["rougeL"] for e in evaluation_results) / len(evaluation_results),
        "bleu": sum(e["bleu"] for e in evaluation_results) / len(evaluation_results),
        "bert_f1": sum(e["bert_f1"] for e in evaluation_results) / len(evaluation_results)
    }
    print("\nAverage Evaluation Metrics:")
    print(json.dumps(avg_metrics, indent=2))

if __name__ == "__main__":
    main()
```

### 使用说明

1. **准备环境**：
   ```bash
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

### 扩展建议

1. **模型切换**：通过修改Config类中的模型路径即可切换不同模型
2. **参数调优**：调整chunk_size、top_k等参数优化检索效果
3. **可视化**：可添加结果可视化模块展示评估指标分布
4. **错误处理**：可扩展更完善的错误处理机制应对不同文件格式的解析异常