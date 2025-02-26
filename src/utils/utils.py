from config.config import Config
from generate.generator import OpenAIGenerator, ZhipuGenerator, VLLMGenerator
from process.rewriter import Rewriter
from eval.evaluator import GenerationEvaluator, LLMJudge, RetrievalEvaluator
from process.rewriter import Rewriter
from process.processor import QuestionGenerator

def create_generator(model_type, config, **kwargs):
    from generate.prompt_builder import LegalPromptBuilder
    from generate.generator import (
        OpenAIGenerator,
        ZhipuGenerator,
        VLLMGenerator,
        HuggingFaceGenerator,
        LocalGenerator
    )

    prompt_builder = kwargs.pop("prompt_builder", LegalPromptBuilder())
    common_params = {
        "config": config,
        "prompt_builder": prompt_builder,
        "max_retries": kwargs.get("max_retries"),
        "max_parallel": kwargs.get("max_parallel"),
        "top_n": kwargs.get("top_n"),
        "batch_size": kwargs.get("batch_size")
    }

    if model_type in ["openai", "qwen", "llama"]:
        return OpenAIGenerator(**common_params)
    elif model_type == "zhipu":
        return ZhipuGenerator(**common_params)
    elif model_type == "vllm":
        return VLLMGenerator(**common_params)
    elif model_type == "huggingface":
        return HuggingFaceGenerator(**common_params)
    elif model_type == "local":
        return LocalGenerator(**common_params)
    elif "model_path" in config:  # Customised model paths
        if "huggingface.co" in config["model_path"]:
            return HuggingFaceGenerator(**common_params)
        else:
            return LocalGenerator(**common_params)
    else:
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

def create_retriever(config=None):
    from retrieval.run_retrieval import Pipeline as RetrievalPipeline
    return RetrievalPipeline(config=config)