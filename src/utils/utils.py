from config.config import Config
from generate.generator import OpenAIGenerator, ZhipuGenerator, VLLMGenerator
from process.rewriter import Rewriter
from eval.evaluator import GenerationEvaluator, LLMJudge, RetrievalEvaluator
from process.rewriter import Rewriter
from process.processor import QuestionGenerator

def create_generator(model_type: str, **kwargs):
    config = CONFIG[model_type]
    if model_type in ["openai", "qwen", "llama"]:
        return OpenAIGenerator(config, **kwargs)
    elif model_type == "zhipu":
        return ZhipuGenerator(config, **kwargs)
    elif model_type == "vllm":
        return VLLMGenerator(config, **kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")

def create_processor(process_type: str, **kwargs):
    if process_type == "rewrite_question":
        required_params = ['config', 'max_retries', 'max_parallel', 'batch_size']
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")            
        return Rewriter(
            config=kwargs['config'],
            max_retries=kwargs['max_retries'],
            max_parallel=kwargs['max_parallel'],
            batch_size=kwargs['batch_size']
        )
    elif process_type in ["current_question", "prefix_question", "prefix_question_answer", "suffix_question"]:
        return QuestionGenerator(process_type, **kwargs)
    else:
        raise ValueError(f"Unsupported processor type: {process_type}")

def create_evaluator(eval_type: str, config):
    if eval_type == "generation":
        return GenerationEvaluator(config)
    elif eval_type == "llm_judge":
        return LLMJudge(config)
    elif eval_type == "retrieval":
        return RetrievalEvaluator(config)
    raise ValueError(f"Unsupported evaluator type: {eval_type}")

def create_retriever():
    from retrieval.run_retrieval import Pipeline as RetrievalPipeline
    return RetrievalPipeline()
