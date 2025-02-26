class Config:
    _default_configs = {
        "openai": {
            "model_type": "openai",
            "model_name": "gpt-3.5-turbo",
            "api_base": "",
            "api_key": "",
            "max_retries": 10,
            "max_parallel": 32
        },
        "zhipu": {
            "model_type": "zhipu",
            "model_name": "glm-4-flash",
            "api_key": "",
            "max_retries": 10,
            "max_parallel": 32
        },
        "llama": {
            "model_type": "llama",
            "model_name": "llama-3.3-70b-instruct",
            "api_base": "",
            "api_key": "",
            "max_retries": 10,
            "max_parallel": 32
        },
        "qwen": {
            "model_type": "qwen",
            "model_name": "qwen2.5-72b-instruct",
            "api_base": "",
            "api_key": "",
            "max_retries": 10,
            "max_parallel": 32
        }
    }

    def __init__(self, model_type=None, my_config=None):
        if my_config:
            self.config = my_config
        elif model_type:
            self.config = self._default_configs.get(model_type)
            if not self.config:
                raise ValueError(f"Invalid model_type: {model_type}")
        else:
            raise ValueError("Must provide either model_type or my_config")

    def get(self, key, default=None):
        return self.config.get(key, default)

    @property
    def model_type(self):
        return self.config["model_type"]
    