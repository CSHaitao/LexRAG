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
