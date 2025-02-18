# LexRAG
Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal Consultation Conversation

## Processor
```
pipeline = ProcessorPipeline("model_type")
pipeline.run_processor(
    original_data_path="data/dataset.json",
    output_path="output_path",
    max_retries=5,
    max_parallel=16,
    batch_size=20
    )
```

## Retriever
### Dense Retrieval
For BGE-base-zh, Qwen2-1.5B, openai
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    retrieval_type="dense",
    model_type="BGE-base-zh",
    question_file_path="question_file_path",
    law_path="data/law_library.jsonl"
    )
```
### Sparse Retrieval
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    retrieval_type="sparse",
    model_type="bm25",
    question_file_path="question_file_path",
    law_path="data/law_library.jsonl"
    )
```

## Generator
```
pipeline = GeneratorPipeline("model_type")
pipeline.run_generator(
    raw_data_path="data/dataset.json",
    retrieval_data_path="retrieval_data_path",
    max_retries=10,
    max_parallel=8,
    top_n=5
    )
```

## Evaluator
### Generation Evaluator
```
pipeline = EvaluatorPipeline()
pipeline.run_evaluator(
    eval_type="generation",
    metrics=["bleu", "rouge", "bert-score"],
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
    metrics=["recall", "ndcg", "mrr"],
    k_values=[1, 3, 5]
    )
```
