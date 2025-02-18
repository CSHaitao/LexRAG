CONFIG = {
    "openai": {
        "model_type": "openai",
        "model_name": "gpt-3.5-turbo",
        "api_base": "base_url",
        "api_key": "api_key",
        "max_retries": 10,
        "max_parallel": 32
    },
    "zhipu": {
        "model_type": "zhipu",
        "model_name": "glm-4-flash",
        "api_key": "api_key",
        "max_retries": 10,
        "max_parallel": 32
    },
    "llama": {
        "model_type": "llama",
        "model_name": "llama-3.3-70b-instruct",
        "api_base": "base_url",
        "api_key": "api_key",
        "max_retries": 10,
        "max_parallel": 32
    },
    "qwen": {
        "model_type": "qwen",
        "model_name": "qwen2.5-72b-instruct",
        "api_base": "base_url",
        "api_key": "api_key",
        "max_retries": 10,
        "max_parallel": 32
    },
    "vllm": {
        "model_type": "vllm",
        "model_path": "/path/to/model",
        "gpu_num": 2
    }
}