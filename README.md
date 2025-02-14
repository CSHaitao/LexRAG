# Retrieval
## Dense Retrieval
BGE/openai/Qwen2-1.5B模型
```
python src/retrieval/run_retrieval.py \
    --model BGE-base-zh \
    --data-path data/law_library.jsonl \
    --question-dir questions \
    --output-dir results/BGE \
    --api-key api_key \
    --base-url base_url \
    --dataset-path data/dataset.json
```
## Sparse Retrieval
BM25
```
python src/retrieval/run_retrieval.py \
    --model bm25 \
    --data-path data/law_library.jsonl \
    --question-dir questions \
    --output-dir results/bm25 \
    --api-key api_key \
    --base-url base_url \
    --dataset-path data/dataset.json
```
:o:Retrieval模块包含Retrieval Evaluator

# Generator
由检索返回的结果，选取top_n作为参考法条
```
python src/generate/run_generate.py \
  --raw_data_path data/dataset.json \
  --data_path data/samples/llm_question_Qwen2-1.5B.jsonl \
  --api_key api_key \
  --base_url base_url \
  --model_name your_model_name \
  --max_retries 10 \
  --max_parallel 32 \
  --top_n 5
```
data_path是检索返回的结果  
model_name是选用的llm(Closed-source LLMs/Open-source LLMs)

# Processor
## rewrite query
```
python src/generate/rewriter.py \
  --model gpt-3.5-turbo \
  --base_url base_url \
  --api_key api_key \
  --max_retries 10 \
  --rpm_limit 3000 \
  --max_parallel 32 \
  --original_data data/dataset.json \
  --output_path question/llm_question.jsonl \
  --batch_size 20
```
model：rewrite所用的模型

# Evaluator
## llm as judge
:arrow_up_small:构造llm as judge所需的prompt  
```
python src/eval/llm_as_judge/make_prompt.py \
  --data_path data/dataset.json \
  --model model_name \
  --version version
```
model和version与输入文件路径相关  
:arrow_up_small:Judge
```
python src/eval/llm_as_judge/run_judge.py \
  --model model_name \
  --version version \
  --base_url base_url \
  --api_key api_key \
  --llm your_model \
  --max_retries 10 \
  --max_parallel 32
```
llm：用于judge的模型

## Generation Evaluator
(多轮对话)对于llm生成的response，评估计算各项指标  
```
python src/eval/run_eval.py \
  --output_dir data/generated_samples \
  --response_file data/samples/generated_responses.jsonl
```  
output_dir输入原数据集[参考答案]  
response_file是大模型生成结果
