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
    doc_dir = "/data/liangrw/test-rag-deepseek-一次性/规范文件"  # 文档目录
    test_data_path = "./test_data.json"  # 测试集路径
    embedding_model_path = "/data/liangrw/models/bge-m3"
    reranker_model_path = "/data/liangrw/models/bge-reranker-large"  # 本地reranker模型路径
    llm_path = "/data/liangrw/models/Qwen2.5-7B-Instruct"  # 本地大模型路径

    # 输出路径
    output_dir = "./output"
    chunk_path = "./output/chunks.json"
    faiss_path = "./output/faiss_index"
    retrieval_path = "./output/retrieval_results.json"
    rerank_path = "./output/rerank_results.json"
    answer_path = "./output/generated_answers.json"
    eval_path = "./output/evaluation_results.json"

    # 参数设置
    chunk_size = 300  # 文本分块长度
    chunk_overlap = 100  # 分块重叠长度
    top_k = 5  # 首次检索数量
    rerank_top_k = 3  # 重排序后保留数量


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
                # loader = PyPDFLoader(file_path, extract_images=True)
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
            "recall_rerank": recall_rerank,
            "relevant_chunks": relevant_chunks
        })

    with open(config.rerank_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# 5.生成答案
def generate_answers(reranked_results):
    print("Generating answers...")

    # 加载本地大模型
    model_name = config.llm_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    results = []
    for item in tqdm(reranked_results):
        # context = "\n\n".join(item["reranked_chunks"])
        
        # 修改连接方式，每个 reranked_chunks 前添加 “context_i：”
        context = "\n\n".join([f"<context_{i + 1}>: {chunk} </context_{i + 1}>" for i, chunk in enumerate(item["reranked_chunks"])])


        prompt = f"根据以下上下文回答问题，如果上下文与问题无关，请回答‘根据上下文不能作出回答’。\n上下文：\n {context} \n\n问题：{item['question']}\n\n答案："
        messages = [
            {"role": "system", "content": "你是一个智能助手，能够帮助回答用户问题。"},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        try:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.6
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            AI_answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            AI_answer = f"生成错误：{str(e)}"

        results.append({
            "question": item["question"],
            "prompt": prompt,
            "generated_answer": AI_answer,
        })

    with open(config.answer_path, 'w', encoding='utf-8') as f:
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
    print("*******************分块完成************************")

    # 2. 创建向量库
    vectorstore = create_vector_store(chunks)
    print("*********************向量库已创建**************************")

    # 加载测试数据
    with open(config.test_data_path) as f:
        test_data = json.load(f)

    # 3. 首次检索
    retrieval_results = first_retrieval(test_data, vectorstore)
    print("*******************retrieval完成************************")

    # 4. 重排序
    reranked_results = rerank_results(retrieval_results)
    print("*******************reranker完成************************")

    # 5. 生成答案
    generated_answers = generate_answers(reranked_results)
    print("*******************LLM回答完成************************")

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

    print("*********************评估完成************************")


if __name__ == "__main__":
    main()