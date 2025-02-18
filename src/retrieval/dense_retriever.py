from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
from openai import OpenAI
import httpx
import faiss
from tqdm import tqdm
import numpy as np
from config.config import CONFIG

class EmbeddingModel:
    def __init__(self, api_key=None, base_url=None):
        if api_key is None or base_url is None:
            self.api_key = CONFIG["openai"]["api_key"]
            self.base_url = CONFIG["openai"]["api_base"]
        else:
            self.api_key = api_key
            self.base_url = base_url

    def load_model(self, model_name):
        if model_name == "BGE-base-zh":
            model_dir = snapshot_download(
                "AI-ModelScope/bge-base-zh-v1.5", revision="master"
            )
            self.model = SentenceTransformer(model_dir, trust_remote_code=True)
        elif model_name == "Qwen2-1.5B":
            model_dir = snapshot_download("iic/gte_Qwen2-1.5B-instruct")
            self.model = SentenceTransformer(model_dir, trust_remote_code=True)
        elif model_name == "openai":
            self.model = None

    def _BGE_embedding(self, texts: list):
        embeddings = self.model.encode(texts)
        return embeddings

    def _Qwen2_embedding(self, texts: list):
        embeddings = self.model.encode(texts)
        return embeddings

    def _openai_embedding(self, texts: list):
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.Client(
                base_url=self.base_url,
                follow_redirects=True,
            ),
        )
        model = "text-embedding-3-small"

        response = client.embeddings.create(
            input=texts,
            model=model,
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings

    def embed(self, text: str, model_name, batch_size=8):
        embed_method = {
            "BGE-base-zh": self._BGE_embedding,
            "Qwen2-1.5B": self._Qwen2_embedding,
            "openai": self._openai_embedding,
        }

        self.load_model(model_name)

        embeddings = []
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i : i + batch_size]
            embeddings.extend(embed_method[model_name](batch))
        embeddings = np.array(embeddings)
        return embeddings

    @staticmethod
    def save_faiss(embeddings, save_path="index.faiss"):
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, save_path)