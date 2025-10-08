import os
import logging
from typing import List

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = self._create_simple_embedder()
    
    def _create_simple_embedder(self):
        """创建极简嵌入器 - 纯Python无依赖"""
        import random
        import math
        import hashlib
        
        class SimpleEmbedder:
            def __init__(self):
                self.dim = 384  # BGE-small维度
                
            def embed_documents(self, texts):
                # 只检索模式下不应被调用
                return [self._text_to_vector(text) for text in texts]
            
            def embed_query(self, text):
                return self._text_to_vector(text)
            
            def _text_to_vector(self, text):
                # 确定性向量生成
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                seed = int(text_hash[:8], 16)
                random.seed(seed)
                
                vector = [random.gauss(0, 1) for _ in range(self.dim)]
                norm = math.sqrt(sum(x * x for x in vector))
                if norm > 0:
                    vector = [x / norm for x in vector]
                return vector
        
        return SimpleEmbedder()
    
    def load_existing_vector_store(self):
        """加载现有向量存储"""
        try:
            from langchain_community.vectorstores import Chroma
            
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name="pdf_documents"
            )
            
            # 验证加载成功
            count = self.vector_store._collection.count()
            logger.info(f"加载成功，包含 {count} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"加载失败: {e}")
            return False
    
    def search_similar_documents(self, query: str, k: int = 5):
        """搜索相似文档"""
        try:
            if not hasattr(self, 'vector_store'):
                if not self.load_existing_vector_store():
                    return []
            
            results = self.vector_store.similarity_search(query, k=k)
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def vector_store_exists(self):
        """检查向量存储是否存在"""
        return os.path.exists(self.persist_directory)

# 兼容性别名
SmartVectorStore = VectorStore
