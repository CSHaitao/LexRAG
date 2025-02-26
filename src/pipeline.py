from utils.utils import create_generator, create_processor, create_evaluator, create_retriever
from generate.data_processor import DataProcessor
from config.config import Config
import json
import os

class GeneratorPipeline:
    def __init__(self, model_type=None, config=None, prompt_builder=None):
        self.model_type = model_type
        from config.config import Config
        if isinstance(config, Config):
            self.config = config.config
        elif config:
            self.config = config
        else:
            cfg = Config(model_type=model_type)
            self.config = cfg.config
            
        from generate.prompt_builder import LegalPromptBuilder
        if prompt_builder:
            self.prompt_builder = prompt_builder
        else:
            self.prompt_builder = LegalPromptBuilder()
        
        self.data_processor = DataProcessor()

    def run_generator(self,
                    raw_data_path,
                    retrieval_data_path,
                    top_n,
                    max_retries=None,
                    max_parallel=None,
                    batch_size=None
                    ):
        processed_data = self.data_processor.process_conversation_turns(raw_data_path)
        retrieval_data = self._load_retrieval_data(retrieval_data_path)
        
        generator = create_generator(
            model_type=self.model_type,
            config=self.config,
            prompt_builder=self.prompt_builder,
            max_retries=max_retries,
            max_parallel=max_parallel,
            top_n=top_n,
            batch_size=batch_size
        )
        generator.generate(processed_data, retrieval_data)
    
    def _load_retrieval_data(self, data_path):
        llm_data = {}
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                llm_data[entry["id"]] = entry
        return llm_data
    
class ProcessorPipeline:

    def __init__(self, model_type=None, config=None):
        if config:
            if isinstance(config, dict):
                self.config = Config(config_dict=config)
            else:
                self.config = config
        elif model_type:
            self.config = Config(model_type=model_type)

    def run_processor(self, 
                     process_type: str,
                     original_data_path: str, 
                     output_path: str,
                     max_retries: int = None,
                     max_parallel: int = None,
                     batch_size: int = None):
        
        if process_type == "rewrite_question":
            processor = create_processor(
                "rewrite_question",
                config=self.config,
                max_retries=max_retries,
                max_parallel=max_parallel,
                batch_size=batch_size
            )
            processor.batch_process(original_data_path, output_path)
        else:
            processor = create_processor(
                process_type=process_type
            )
            processor.run_process(original_data_path, output_path)

class EvaluatorPipeline:
    def __init__(self, model_type: str = None):
        self.model_type = model_type
        self.config = Config._default_configs[model_type] if model_type else None

    def run_evaluator(self, 
                    eval_type,
                    metrics=None,
                    data_path: str = None,
                    gen_path: str = None,
                    response_file: str = None,
                    results_path: str = None,
                    k_values=None):
        evaluator = create_evaluator(eval_type, self.config)
        
        if eval_type == "generation":
            return evaluator.evaluate(data_path, response_file, metrics)
        elif eval_type == "llm_judge":
            return evaluator.evaluate(data_path, gen_path)
        elif eval_type == "retrieval":
            return evaluator.evaluate(results_path, metrics, k_values)
        raise ValueError(f"Unsupported evaluator type: {eval_type}")

class RetrieverPipeline:
    def __init__(self, config=None):
        self.config = config
        self.retriever = create_retriever(config=self.config)

    def run_retriever(self, model_type, question_file_path, law_path, **kwargs):
        self.retriever.run_retriever(
            model_type=model_type,
            question_file_path=question_file_path,
            law_path=law_path,
            **kwargs
        )
    