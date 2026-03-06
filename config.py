import os
from typing import List
from dotenv import load_dotenv


class Config:
    """
    配置中心 - 统一管理项目所有配置项
    """

    def __init__(self):
        load_dotenv()
        
        self.API_KEY = os.getenv("DEEPSEEK_API_KEY")
        if not self.API_KEY:
            raise ValueError("未找到 DEEPSEEK_API_KEY！请检查 .env 文件是否配置正确。")
        
        self.API_BASE_URL = "https://api.deepseek.com"
        self.MODEL_NAME = "deepseek-chat"
        
        self.EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
        
        self.USE_MODEL_SCOPE = os.getenv("USE_MODEL_SCOPE", "true").lower() == "true"
        self.MODEL_SCOPE_CACHE_DIR = os.getenv("MODEL_SCOPE_CACHE_DIR", "./models")
        self.USE_UNSTRUCTURED = os.getenv("USE_UNSTRUCTURED", "false").lower() == "true"
        
        self.CHUNK_SIZE = 800
        self.CHUNK_OVERLAP = 200
        
        self.VECTOR_DB_PATH = "./chroma_db"
        self.DATA_DIR = "./data"
        self.FILE_HASH_PATH = os.path.join(self.VECTOR_DB_PATH, "file_hashes.json")
        
        self.SUPPORTED_EXTENSIONS = [".docx", ".pdf", ".txt"]
        
        self.RETRIEVE_TOP_K = 5
        self.TEMPERATURE = 0.1


config = Config()
