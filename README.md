# LexRAG
Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal Consultation Conversation

## Processor
Rewrite query
```
pipeline = ProcessorPipeline(model_type="")
pipeline.run_processor(
    process_type="rewrite_question",
    original_data_path="data/dataset.json",
    output_path="data/rewrite_question.jsonl",
    max_retries=5,
    max_parallel=32,
    batch_size=20
)
```
Other different type of constructing the query, including using the last question, the entire conversation context, or the entire query history
Take ‘using the last question’ as an example.
```
pipeline = ProcessorPipeline()
pipeline.run_processor(
    process_type="current_question",
    original_data_path="data/dataset.json",
    output_path="data/current_question.jsonl"
)
```

## Retriever
### Dense Retrieval
For BGE-base-zh, Qwen2-1.5B, openai(openai can support multiple models)
```
openai_config = {
    "api_key": "your_api_key",
    "base_url": "your_base_url"
}
pipeline = RetrieverPipeline(config=openai_config)
pipeline.run_retriever(
    model_type="openai",
    model_name="text-embedding-3-small",
    faiss_type="FlatIP",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
    )
```
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    model_type="Qwen2-1.5B",
    faiss_type="FlatIP",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
    )
```
Can support three mainstream faiss types: FlatIP, HNSW and IVF.
### Sparse Retrieval
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    model_type="bm25",
    bm25_backend="bm25s",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
)
```
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    model_type="bm25",
    bm25_backend="pyserini",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
)
```
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    model_type="qld",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
)
```
Support for QLD implemented by the ```pyserini``` library and BM25 implemented by ```bm25s``` or ```pyserini```.

## Generator
```
pipeline = GeneratorPipeline(model_type="")
pipeline.run_generator(
    raw_data_path="data/dataset.json",
    retrieval_data_path="data/samples/llm_question_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
Support many common llm models, just enter the model name in model_type (for common models, you need to modify the corresponding api_key and base_url in ```config.py```)   
```
custom_config = {
    "model_type": "vllm",
    "model_path": "lmsys/vicuna-7b-v1.3",
    "gpu_num": 2
}
pipeline = GeneratorPipeline(model_type="vllm", config=custom_config)
pipeline.run_generator(
    raw_data_path="data/dataset.json",
    retrieval_data_path="data/samples/llm_question_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
```
hf_config = {
    "model_type": "huggingface",
    "model_path": "hf_model_path"
}
pipeline = GeneratorPipeline(
    model_type="huggingface",
    config=hf_config,
)
pipeline.run_generator(
    raw_data_path="data/dataset.json",
    retrieval_data_path="data/samples/llm_question_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
```
local_config = {
    "model_type": "local",
    "model_path": "local_model_path"
}
pipeline = GeneratorPipeline(
    model_type="local",
    config=local_config,
)
pipeline.run_generator(
    raw_data_path="data/dataset.json",
    retrieval_data_path="data/samples/llm_question_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
Support for response generation using ```vllm```, ```huggingface``` and local models.    
```
from generate.prompt_builder import LegalPromptBuilder, CustomSystemPromptBuilder, FullCustomPromptBuilder
custom_prompt = CustomSystemPromptBuilder("请用一句话回答法律问题：")
pipeline = GeneratorPipeline(
    model_type="",
    prompt_builder=custom_prompt
)
pipeline.run_generator(
    raw_data_path="data/dataset.json",
    retrieval_data_path="data/samples/llm_question_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
```
def full_custom_builder(history, question, articles):
    return [
        {"role": "user", "content": f"请以“回答如下：”为开头回答\n问题：{question}（相关法条：{','.join(articles)}）"}
    ]

pipeline = GeneratorPipeline(
    model_type="",
    prompt_builder=FullCustomPromptBuilder(full_custom_builder)
)
pipeline.run_generator(
    raw_data_path="data/dataset.json",
    retrieval_data_path="data/samples/llm_question_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
Supports customisation of the input prompt. By default we use our defined ```LegalPromptBuilder```, users can choose to use ```CustomSystemPromptBuilder``` to customise the ```system``` content, or they can choose to use ```FullCustomPromptBuilder``` for full prompt customisation.

## Evaluator
### Generation Evaluator
```
pipeline = EvaluatorPipeline()
pipeline.run_evaluator(
    eval_type="generation",
    metrics=["bleu", "rouge", "bert-score", "keyword_accuracy", "char_scores", "meteor"],
    data_path="data/dataset.json",
    response_file="response_file_path"
    )
```
### LLM-as-a-Judge
```
pipeline = EvaluatorPipeline("model_type")
pipeline.run_evaluator(
    eval_type="llm_judge",
    data_path="data/dataset.json",
    gen_path="generated_responses_path"
    )
```
### Retrieval Evaluator
```
pipeline = EvaluatorPipeline()
pipeline.run_evaluator(
    eval_type="retrieval",
    results_path="retrieval_results_path",
    metrics=["recall", "precision", "f1", "ndcg", "mrr"],
    k_values=[1, 3, 5]
    )
```
