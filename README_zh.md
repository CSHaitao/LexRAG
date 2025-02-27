 <div align=center>
<img src="https://github.com/user-attachments/assets/3b8c841d-694a-49d2-9629-dd3cbc4f649b" width="210px">
</div>

<h1 align="center">LexRAG: 多轮法律咨询对话中的检索-增强生成基准测试</h1>
<p align="center">
  :book:<a href="./README_zh.md">中文</a> | 
  <a href="./README.md">English</a>
</p>
欢迎来到LexiT，一个法律领域的检索增强生成（RAG）专用工具包。

## :link: Introduction
为了推进法律领域的 RAG 系统研究，我们为法律研究人员提出了模块化、可扩展的 RAG 工具包 LexiT。虽然目前已有一些通用领域的 RAG 工具包，但它们并不支持多轮对话和针对法律领域的评估。LexiT 由三个部分组成： ```Data``` ```Pipeline``` ```Evaluation```。LexiT将 RAG 流程的所有要素整合到一个统一的框架中，并支持独立应用。这种模块化设计提高了灵活性，在评估不同的法律情况时具有高度的可定制性。
 <div align=center>
<img src="https://github.com/user-attachments/assets/b2badd1e-55a3-42d8-ae10-758e5f1ae6f0" width="500px">
</div>

## :books: Data
* 数据组件由两个关键要素组成：输入对话和语料库   
  * 对话格式可以是单轮对话，也可以是多轮对话。多轮对话提供以前的对话历史作为背景。   
    我们提供一个确保准确性和专业性的数据集 ```./data/dataset.json``` ，包含 1,013 个多轮对话，每个对话有 5 轮问题和回答。   
  * 对于语料库 ```./data/law_library.jsonl``` ，我们从三个不同来源收集原始数据。除了作为本文候选语料库的法律条文外，为方便研究人员使用，工具包中还包括法律文书和法律案例。具体来说，法律条文包含了中国各种成文法中的 17,228 个条文。   
 <div align=center>
<img src="https://github.com/user-attachments/assets/5464a404-98c6-45b6-90a8-65b936824cf1" width="350px">
</div>

## :rocket: Pipeline
### :bookmark_tabs: Processor
Processor将对话转换成Retriever使用的查询。我们支持几种构建查询的策略，包括使用最后一个问题、整个对话上下文或整个查询历史等。 运行 ```./src/pipeline.py``` :
```
pipeline = ProcessorPipeline()
pipeline.run_processor(
    process_type="process_type",
    original_data_path="data/dataset.json",
    output_path="data/current_question.jsonl"
)
```
```--process_type```: ```current_question``` ```prefix_question``` ```prefix_question_answer``` ```suffix_question``` 构建查询的策略   
```--original_data_path```: 需要进行处理的对话数据路径   
```--output_path```: 输出路径   

此外，我们还预定义了一种查询重写策略，利用LLM将所有必要的上下文整合为一个清晰、独立的问题。
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
```--model_type```: 我们在 ```./src/config/config.py``` 中提供了一些默认模型，可以通过更改模型的配置信息进行使用。如果您想使用其他模型，可以将 ```model_type=""``` 替换为 ```config=``` ，并自定义配置信息。   
```--max_retries``` ```--max_parallel```: 并行处理参数   
```--batch_size```: 批次大小   

您可以在 ```output_path```查看输出结果。 以重写查询策略输出结果 ```./data/samples/rewrite_question.jsonl``` 为例，您可以在 ```"question"``` 中查看处理后查询。   

### :bookmark_tabs: Retriever
#### Dense Retrieval
我们支持 BGE 和 GTE 等高级模型，您可以使用本地加载的模型或 API 调用对向量进行编码。使用```Faiss```进行索引构建，支持三种faiss类型：```FlatIP```、```HNSW```和```IVF```。   
* 对于API调用   

运行 ```./src/pipeline.py``` :
```
openai_config = {
    "api_key": "your_api_key",
    "base_url": "your_base_url"
}
pipeline = RetrieverPipeline(config=openai_config)
pipeline.run_retriever(
    model_type="openai",
    model_name="model_name",
    faiss_type="faiss_type",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
    )
```
```--model_name```: embedding模型名称   
```--question_file_path```: 处理后文件路径 (by *Processor*)   
```--law_path```: 语料库路径   

* 对于加载的模型
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    model_type="BGE-base-zh",
    faiss_type="faiss_type",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
    )
```
```--model_type```: embedding模型名称

#### Sparse Retrieval
对于词法匹配，我们使用 ```Pyserini``` 库实现 ```BM25``` 和 ```QLD``` ，同时支持使用 ```bm25s``` 实现 ```BM25``` 。   
* 对于BM25:
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    model_type="bm25",
    bm25_backend="bm25_backend",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
)
```
```--bm25_backend```: ```bm25s``` ```pyserini``` 构建bm25索引的方法   
```--question_file_path```: 处理后文件路径 (by *Processor*)   
```--law_path```: 语料库路径

* 对于QLQ:
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    model_type="qld",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
)
```

> 使用Dense Retrieval, 您可以在 ```./data/retrieval/law_index_{model_type}.faiss``` 查看索引并在 ```./data/retrieval/res/retrieval_{model_type}.jsonl``` 查看输出检索结果。 以GTE_Qwen2-1.5B模型输出结果 ```./data/samples/retrieval_Qwen2-1.5B.jsonl``` 为例，您可以在 ```"recall"``` 查看检索召回结果。
    
> 使用Sparse Retrieval, 您可以在 ```./data/retrieval/pyserini_index```(pyserini) 查看索引并在 ```./data/retrieval/res/retrieval_{model_type}_{bm25_backend}.jsonl``` 查看输出检索结果。

### :bookmark_tabs: Generator
我们支持主流的LLMs进行回答生成。 运行 ```./src/pipeline.py``` :
```
pipeline = GeneratorPipeline(model_type="")
pipeline.run_generator(
    raw_data_path="data/dataset.json",
    retrieval_data_path="data/samples/retrieval_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
```--model_type```: 支持常用的LLMs，只需要在 ```model_type``` 输入模型名称（常用模型已在 ```./src/config/config.py``` 中设置，您修改相关配置信息即可使用）   
```--raw_data_path```: 包含问题的对话数据路径   
```--retrieval_data_path```: 检索得到的数据路径   
```--max_retries``` ```--max_parallel```: 并行处理相关参数   
```--batch_size```: 批次大小   
```--top_n```: 使用检索结果中的top_n法条作为回答参考


我们还支持使用 ```vllm``` ```huggingface``` 和本地模型进行回答生成。   
* 对于 ```vllm```
```
custom_config = {
    "model_type": "vllm",
    "model_path": "vllm_model_path",
    "gpu_num": 2
}
pipeline = GeneratorPipeline(model_type="vllm", config=custom_config)
pipeline.run_generator(
    raw_data_path="data/dataset.json",
    retrieval_data_path="data/samples/retrieval_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
* 对于 ```huggingface```
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
    retrieval_data_path="data/samples/retrieval_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
* 对于本地模型
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
    retrieval_data_path="data/samples/retrieval_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```   


我们支持使用灵活自定义的prompt构造。 默认情况使用我们预定义的 ```LegalPromptBuilder``` 进行prompt构造，您还可以选择使用 ```CustomSystemPromptBuilder``` 自定义prompt的系统角色部分，或者选择 ```FullCustomPromptBuilder``` 完全自定义prompt。   
* 对于 ```CustomSystemPromptBuilder```:
```
from generate.prompt_builder import LegalPromptBuilder, CustomSystemPromptBuilder, FullCustomPromptBuilder
custom_prompt = CustomSystemPromptBuilder("请用一句话回答法律问题：")
pipeline = GeneratorPipeline(
    model_type="",
    prompt_builder=custom_prompt
)
pipeline.run_generator(
    raw_data_path="data/dataset.json",
    retrieval_data_path="data/samples/retrieval_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
```--prompt_builder```: 您可以使用 ```CustomSystemPromptBuilder(" ")``` 自定义系统角色   

* 对于 ```FullCustomPromptBuilder```:
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
    retrieval_data_path="data/samples/retrieval_Qwen2-1.5B.jsonl",
    max_retries=5,
    max_parallel=32,
    top_n=5,
    batch_size=20
)
```
```--prompt_builder```: ```FullCustomPromptBuilder``` 支持输入 ```history```, ```question```, ```articles```, 您可以先自定义prompt策略，再作为prompt_builder输入。   
> ```history```: 对话历史   
> ```question```: 多轮对话的当前问题   
> ```articles```: 检索返回结果的 ```--top_n``` 参考法条   

您可以在 ```./data/generated_responses.jsonl``` 查看输出。 示例输出 ```./data/samples/generated_responses.jsonl```。 ```.jsonl``` 文件格式如下:
```
{"id": "xxx", "question": "...", "response": "..."}
```

## :pencil: Evaluation
### Generation Evaluator
生成评估器衡量生成回答与参考答案之间的一致性，支持指标如ROUGE, BLEU, METEOR, BERTScore。 运行 ```./src/pipeline.py```:
```
pipeline = EvaluatorPipeline()
pipeline.run_evaluator(
    eval_type="generation",
    metrics=["bleu", "rouge", "bert-score", "keyword_accuracy", "char_scores", "meteor"],
    data_path="data/dataset.json",
    response_file="response_file_path"
    )
```
```--data_path```: 原始包含查询问题的数据集路径   
```--response_file```: LLM生成回答的路径   

### Retrieval Evaluator
检索评估器评估检索文件的相关性和准确性，支持主流指标如NDCG, Recall, MRR, Precision, F1的计算。   
```
pipeline = EvaluatorPipeline()
pipeline.run_evaluator(
    eval_type="retrieval",
    results_path="retrieval_results_path",
    metrics=["recall", "precision", "f1", "ndcg", "mrr"],
    k_values=[1, 3, 5]
    )
```
```--results_path```: 检索结果路径   
```--k_values```: 考虑分数最高的k个结果   
> 您可以在 ```./data/retrieval/report.jsonl``` 查看结果。   

### LLM-as-a-Judge
LLM通过多维思维链推理来评估回答质量。 LLM-as-a-Judge 使用的prompt为 ```./src/config/template/prompt.txt```
```
pipeline = EvaluatorPipeline("model_type")
pipeline.run_evaluator(
    eval_type="llm_judge",
    data_path="data/dataset.json",
    gen_path="generated_responses_path"
    )
```
```--model_type```: 评估使用的模型名称   
```--data_path```: 原始包含查询问题的数据集路径   
```--gen_path```: LLM生成回答的路径   
> 您可以在 ```./data/results/turn{turn}/judge_results.jsonl``` 查看结果。  
