from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
from openai import OpenAI
import httpx
import faiss
from tqdm import tqdm
import numpy as np

class DenseRetriever:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = None

    def load_model(self, model_type):
        if model_type == "BGE-base-zh":
            model_dir = snapshot_download(
                "AI-ModelScope/bge-base-zh-v1.5", revision="master"
            )
            self.model = SentenceTransformer(model_dir, trust_remote_code=True)
        elif model_type == "Qwen2-1.5B": #GTE model
            model_dir = snapshot_download("iic/gte_Qwen2-1.5B-instruct")
            self.model = SentenceTransformer(model_dir, trust_remote_code=True)
        elif model_type == "openai":
            self.model = None

    def _BGE_embedding(self, texts: list):
        embeddings = self.model.encode(texts)
        return embeddings

    def _Qwen2_embedding(self, texts: list):
        embeddings = self.model.encode(texts)
        return embeddings

    def _openai_embedding(self, texts: list, model_name):
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.Client(
                base_url=self.base_url,
                follow_redirects=True,
            ),
        )

        response = client.embeddings.create(
            input=texts,
            model=model_name,
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings
        
    def embed(self, texts, model_type, model_name=None, batch_size=8):
        if model_type in ["BGE-base-zh", "Qwen2-1.5B"]:
            self.load_model(model_type)
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i : i + batch_size]
                embeddings.extend(self.model.encode(batch))
            return np.array(embeddings)
        elif model_type == "openai":
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i : i + batch_size]
                batch_embeddings = [self._openai_embedding([text], model_name) for text in batch]
                embeddings.extend(batch_embeddings)
            return np.array(embeddings)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def save_faiss(embeddings, faiss_type, save_path="index.faiss"):
        dim = embeddings.shape[1]
        
        if faiss_type == "FlatIP":
            index = faiss.IndexFlatIP(dim)
        elif faiss_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 64)
        elif faiss_type == "IVF":
            nlist = min(128, int(np.sqrt(len(embeddings))))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(embeddings.astype('float32'))
            index.nprobe = min(8, nlist//4)
        else:
            raise ValueError(f"Unsupported FAISS type: {faiss_type}")
        
        index.add(embeddings.astype('float32'))
        faiss.write_index(index, save_path)
