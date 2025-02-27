 <div align=center>
<img src="https://github.com/user-attachments/assets/3b8c841d-694a-49d2-9629-dd3cbc4f649b" width="210px">
</div>

<h1 align="center">LexRAG: Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal Consultation Conversation</h1>
<p align="center">
  :book:<a href="./README_zh.md">中文</a> | 
  <a href="./README.md">English</a>
</p>
Welcome to LexiT, the dedicated toolkit for RAG in the legal domain.

## :link: Introduction
To advance RAG system research in the legal domain, we’ve proposed LexiT, a modular and scalable RAG toolkit for legal researchers. Although there are some general-domain RAG toolkits available, they do not support multi-turn conversations and evaluations tailored to the legal domain. LexiT consists of three components: **Data**, **Pipeline**, and **Evaluation**. It integrates all elements of the RAG process into a unified framework and supports standalone applications. This modular design enhances flexibility and allows for high customizability in evaluating different legal scenarios.
 <div align=center>
<img src="https://github.com/user-attachments/assets/b2badd1e-55a3-42d8-ae10-758e5f1ae6f0" width="500px">
</div>

## :books: Data
* The data component consists of two key elements: input conversations and corpora.   
  * The conversation format can be either single-turn or multi-turn. Multi-turn conversations provide previous dialogue history as context.   
    The dataset ```./data/dataset.json``` contains 1,013 multi-turn conversations, each with 5 rounds of questions and responses.    
  * For the corpora, we collect raw data from three different sources. In addition to Legal Articles, which serve as the candidate corpus in this paper, Legal Books and Legal Cases are also included in the toolkit for researchers’ convenience. Specifically, Legal Articles contains 17,228 provisions from various Chinese statutory laws.   
    The corpus is stored in ```./data/law_library.jsonl```.
 <div align=center>
<img src="https://github.com/user-attachments/assets/5464a404-98c6-45b6-90a8-65b936824cf1" width="350px">
</div>

## :rocket: Pipeline
### :bookmark_tabs: Processor
The processor is responsible for converting the conversation into queries used by the retriever. There are several strategies for constructing the query, including using the last question, the entire conversation context, or the entire query history. Run ```./src/pipeline.py``` :
```
pipeline = ProcessorPipeline()
pipeline.run_processor(
    process_type="process_type",
    original_data_path="data/dataset.json",
    output_path="data/current_question.jsonl"
)
```
```--process_type```: ```current_question``` ```prefix_question``` ```prefix_question_answer``` ```suffix_question``` the strategy for constructing the query   
```--original_data_path```: the path to the conversation data you want to process   
```--output_path```: the path for output   

Moreover, we also predefined a query rewrite strategy, which employs an LLM to integrate all necessary context into a clear, standalone question.
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
```--model_type```: We provide some default models that are stored in ```./src/config/config.py```, which can use by changing the configuration information. If you want to use other models, you can replace ```model_type=""``` with ```config=``` and customise the configuration information.   
```--max_retries``` ```--max_parallel```: parallel processing parameter   
```--batch_size```: batch size   

You can check the results in ```output_path```. A sample processed data is ```./data/samples/rewrite_question.jsonl``` which you can see processed query in ```"question"```.

### :bookmark_tabs: Retriever
#### Dense Retrieval
We support advanced models such as BGE and GTE. Users can encode vectors using locally loaded models or API calls. We employ the ```Faiss``` for index construction which can support three mainstream faiss types: ```FlatIP```, ```HNSW``` and ```IVF```.
* For API calls
Run ```./src/pipeline.py``` :
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
```--model_name```: the model for embedding   
```--question_file_path```: the path for processed queries (by *Processor*)   
```--law_path```: the path for corpus
* For loaded models
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    model_type="BGE-base-zh",
    faiss_type="faiss_type",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
    )
```
```--model_type```: the model for embedding

#### Sparse Retrieval
For lexical matching, we use the ```Pyserini``` library to implement ```BM25``` and ```QLD``` while supporting ```bm25s``` to implement ```BM25```.   
* For BM25:
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    model_type="bm25",
    bm25_backend="bm25_backend",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
)
```
```--bm25_backend```: ```bm25s``` ```pyserini``` the method for building bm25 index   
```--question_file_path```: the path for processed queries (by *Processor*)   
```--law_path```: the path for corpus

* For QLQ:
```
pipeline = RetrieverPipeline()
pipeline.run_retriever(
    model_type="qld",
    question_file_path="data/rewrite_question.jsonl",
    law_path="data/law_library.jsonl"
)
```

> For Dense Retrieval, You can check the index in ```./data/retrieval/law_index_{model_type}.faiss``` and the results in ```./data/retrieval/res/retrieval_{model_type}.jsonl```. A sample retrieve data is ```./data/samples/retrieval_Qwen2-1.5B.jsonl``` which you can see retrieve recall in ```"recall"```.

> For Sparse Retrieval, You can check the index in ```./data/retrieval/pyserini_index```(pyserini) and the results in ```./data/retrieval/res/retrieval_{model_type}_{bm25_backend}.jsonl```.

### :bookmark_tabs: Generator
We support mainstream LLMs for generating responses. Run ```./src/pipeline.py``` :
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
```--model_type```: Support common LLMs, just enter the model name in ```model_type``` (for common models, you need to modify the corresponding configuration information in ```./src/config/config.py```)   
```--raw_data_path```: the path for conversation data which includes queries   
```--retrieval_data_path```: the path for retrieve data   
```--max_retries``` ```--max_parallel```: parallel processing parameter   
```--batch_size```: batch size   
```--top_n```: use the top_n of the retrieved return laws as references


We also support for response generation using ```vllm```, ```huggingface``` and local models.   
* For ```vllm```
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
* For ```huggingface```
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
* For local model
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


We supports flexible customisation of the input prompt. By default we use our defined ```LegalPromptBuilder```, you can choose to use ```CustomSystemPromptBuilder``` to customise the system content, or ```FullCustomPromptBuilder``` for full prompt customisation.   
* For ```CustomSystemPromptBuilder```:
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
```--prompt_builder```: you can use ```CustomSystemPromptBuilder(" ")``` customising the system base used by LLM   

* For ```FullCustomPromptBuilder```:
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
```--prompt_builder```: ```FullCustomPromptBuilder``` supports ```history```, ```question```, ```articles``` as input, you can customize the prompt strategy first, and then used it as prompt_builder.   
> ```history```: conversation history   
> ```question```: current query for LLM   
> ```articles```: the retrieved return ```--top_n``` articles as references   

You can check the results in ```./data/generated_responses.jsonl```. A sample processed data is ```./data/samples/generated_responses.jsonl```. The ```.jsonl``` file format for each line is as follows:
```
{"id": "xxx", "question": "...", "response": "..."}
```

## :pencil: Evaluation
### Generation Evaluator
The generation evaluator measures the consistency between generated responses and reference answers, supporting automated metrics like ROUGE, BLEU, METEOR, and BERTScore. Run ```./src/pipeline.py```:
```
pipeline = EvaluatorPipeline()
pipeline.run_evaluator(
    eval_type="generation",
    metrics=["bleu", "rouge", "bert-score", "keyword_accuracy", "char_scores", "meteor"],
    data_path="data/dataset.json",
    response_file="response_file_path"
    )
```
```--data_path```: the path to original query dataset   
```--response_file```: the path to LLM's generated responses   

### Retrieval Evaluator
The retrieval evaluator assesses the relevance and accuracy of retrieved documents, supporting the calculation of mainstream automated metrics such as NDCG, Recall, MRR, Precision, and F1.
```
pipeline = EvaluatorPipeline()
pipeline.run_evaluator(
    eval_type="retrieval",
    results_path="retrieval_results_path",
    metrics=["recall", "precision", "f1", "ndcg", "mrr"],
    k_values=[1, 3, 5]
    )
```
```--results_path```: the path for retrieval results   
```--k_values```: consider the highest k scores in the ranking   
> You can check the results in ```./data/retrieval/report.jsonl```.   

### LLM-as-a-Judge
LLM judge evaluates response quality through multidimensional chain of thought reasoning. The prompt we used for LLM-as-a-Judge is ```./src/config/template/prompt.txt```
```
pipeline = EvaluatorPipeline("model_type")
pipeline.run_evaluator(
    eval_type="llm_judge",
    data_path="data/dataset.json",
    gen_path="generated_responses_path"
    )
```
```--model_type```: the model as LLM Judge   
```--data_path```: the path to original query dataset   
```--gen_path```: the path to LLM's generated responses   
> You can check the results in ```./data/results/turn{turn}/judge_results.jsonl```.  


