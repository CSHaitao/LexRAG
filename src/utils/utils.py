from config.config import CONFIG
from generate.generator import OpenAIGenerator, ZhipuGenerator, VLLMGenerator
from generate.rewriter import Rewriter
from eval.evaluator import GenerationEvaluator, LLMJudge, RetrievalEvaluator

def create_generator(model_type: str, **kwargs):
    config = CONFIG[model_type]
    if model_type in ["openai", "qwen", "llama"]:
        return OpenAIGenerator(config, **kwargs)
    elif model_type == "zhipu":
        return ZhipuGenerator(config, **kwargs)
    elif model_type == "vllm":
        return VLLMGenerator(config, **kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")

def create_rewriter(model_type: str, **kwargs):
    config = CONFIG[model_type].copy()
    return Rewriter(config, **kwargs)

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