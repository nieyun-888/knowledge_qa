import os
from typing import List

class Config:
    # 路径配置
    CHROMA_DB_PATH = "chroma_db"
    
    # 文本处理配置
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    
    # Chroma DB配置
    COLLECTION_NAME = "pdf_documents"
    
    # DeepSeek API配置
    DEEPSEEK_API_KEY = ""  # 在secrets中配置
    DEEPSEEK_API_BASE = "https://api.lkeap.cloud.tencent.com/v1"
    DEEPSEEK_MODEL = "deepseek-v3.1"
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        os.makedirs(cls.CHROMA_DB_PATH, exist_ok=True)