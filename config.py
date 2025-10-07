import os
from typing import List

class Config:
    # 路径配置
    PDF_DIR = "data/raw_pdfs"
    CHROMA_DB_PATH = "chroma_db"
    PROCESSED_DIR = "data/processed"
    # MODEL_DIR = "models/bge-small-zh-v1.5"
    
    # 文本处理配置
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    
    # Chroma DB配置
    COLLECTION_NAME = "knowledge_base"
    
    # DeepSeek API配置
    DEEPSEEK_API_KEY = ""  # 在secrets中配置
    DEEPSEEK_API_BASE = "https://api.lkeap.cloud.tencent.com/v1"
    DEEPSEEK_MODEL = "deepseek-v3.1"
    
    # 部署配置
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_EXTENSIONS = ['.pdf']
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        os.makedirs(cls.CHROMA_DB_PATH, exist_ok=True)
    
    @classmethod
    def get_model_path(cls):
        """获取模型路径"""
        return "BAAI/bge-small-zh-v1.5"