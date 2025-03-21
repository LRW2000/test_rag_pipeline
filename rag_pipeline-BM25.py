import os
import json
import jieba
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score


# 配置参数
class Config:
    doc_dir = "./docs"
    test_data_path = "./test_data.json"
    reranker_model_path = "/data/liangrw/models/bge-reranker-large"
    llm_path = "/data/liangrw/models/Qwen2.5-7B-Instruct"

    output_dir = "./output"
    chunk_path = "./output/chunks.json"
    retrieval_path = "./output/retrieval_results.json"
    rerank_path = "./output/rerank_results.json"
    answer_path = "./output/generated_answers.json"
    eval_path = "./output/evaluation_results.json"

    chunk_size = 300
    chunk_overlap = 100
    top_k = 5  # BM25 检索数量
    rerank_top_k = 3  # 重排序后保留数量


config = Config()

os.makedirs(config.output_dir, exist_ok=True)

# 1. 加载和分块
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

    chunk_data = [{"text": doc.page_content, "metadata": doc.metadata} for doc in chunks]
    with open(config.chunk_path, 'w') as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    return chunk_data  # 返回文本数据


# 2. 构建 BM25 索引
def build_bm25_index(chunks):
    print("Building BM25 index...")

    # 处理文档
    corpus = [chunk["text"] for chunk in chunks]
    tokenized_corpus = [list(jieba.cut(text)) for text in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus


# 3. BM25 检索
def bm25_retrieval(test_data, bm25, corpus):
    print("Running BM25 retrieval...")
    results = []
    for item in tqdm(test_data):
        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        # 对问题分词
        tokenized_query = list(jieba.cut(question))

        # 计算 BM25 分数
        scores = bm25.get_scores(tokenized_query)

        # 选取 top_k 文档
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:config.top_k]
        retrieved = [corpus[i] for i in top_indices]

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

        pairs = [(question, chunk) for chunk in chunks]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        reranked = [chunk for chunk, _ in ranked[:config.rerank_top_k]]

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
        context = "\n".join(item["reranked_chunks"])
        prompt = f"根据以下上下文回答问题，如果上下文与问题无关，请回答‘根据上下文不能作出回答’。\n上下文：{context}\n问题：{item['question']}\n答案："

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
    chunks = load_and_chunk()
    bm25, corpus = build_bm25_index(chunks)

    with open(config.test_data_path) as f:
        test_data = json.load(f)

    retrieval_results = bm25_retrieval(test_data, bm25, corpus)
    reranked_results = rerank_results(retrieval_results)
    generated_answers = generate_answers(reranked_results)
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
