import os
import logging
from typing import List
import random
import math
import hashlib

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = self._create_simple_embedder()
        logger.info("向量储存初始化完成")
    
    def _create_simple_embedder(self):
        """使用真实BGE模型替代随机向量"""
        try:
            # 首先尝试加载本地模型
            from sentence_transformers import SentenceTransformer
            model_path = "./models/bge-small-zh-v1.5"
            if os.path.exists(model_path):
                logger.info("加载本地BGE-small模型...")
                model = SentenceTransformer(model_path)
                
                class RealEmbedder:
                    def __init__(self, model):
                        self.model = model
                        self.dim = 512  # BGE-small的维度
                    
                    def embed_documents(self, texts):
                        logger.info(f"为 {len(texts)} 个文本生成嵌入")
                        embeddings = self.model.encode(texts, normalize_embeddings=True)
                        return embeddings.tolist()
                    
                    def embed_query(self, text):
                        embedding = self.model.encode([text], normalize_embeddings=True)
                        return embedding[0].tolist()
                
                logger.info("✅ 本地BGE模型加载成功")
                return RealEmbedder(model)
        except Exception as e:
            logger.warning(f"本地模型加载失败: {e}")

        try:
            # 备选：在线下载模型
            from sentence_transformers import SentenceTransformer
            logger.info("在线下载BGE-small模型...")
            model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            
            class RealEmbedder:
                def __init__(self, model):
                    self.model = model
                    self.dim = 512
                
                def embed_documents(self, texts):
                    logger.info(f"为 {len(texts)} 个文本生成嵌入")
                    embeddings = self.model.encode(texts, normalize_embeddings=True)
                    return embeddings.tolist()
                
                def embed_query(self, text):
                    embedding = self.model.encode([text], normalize_embeddings=True)
                    return embedding[0].tolist()
            
            logger.info("✅ 在线BGE模型加载成功")
            return RealEmbedder(model)
            
        except Exception as e:
            logger.error(f"所有真实模型加载失败，使用备用嵌入器: {e}")
            # 回退到随机向量
            return self._create_fallback_embedder()

    def _create_fallback_embedder(self):
        """备用随机向量生成器"""
        class SimpleEmbedder:
            def __init__(self):
                self.dim = 512
                
            def embed_documents(self, texts):
                return [self._text_to_vector(text) for text in texts]
            
            def embed_query(self, text):
                return self._text_to_vector(text)
            
            def _text_to_vector(self, text):
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

    def vector_store_exists(self):
        """检查向量存储是否存在"""
        return os.path.exists(self.persist_directory)

    def search_similar_documents(self, query: str, k: int = 5):
        """搜索相似文档"""
        try:
            logger.info(f"开始搜索查询: {query}")
            
            if not hasattr(self, 'vector_store'):
                logger.info("向量存储未加载，正在加载...")
                if not self.load_existing_vector_store():
                    logger.error("向量存储加载失败")
                    return []
            
            logger.info("正在执行相似度搜索...")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"搜索完成，找到 {len(results)} 个相关文档")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_document_stats(self):
        """获取文档统计信息 - 用于前端显示"""
        try:
            if not hasattr(self, 'vector_store'):
                if not self.load_existing_vector_store():
                    return {}
            
            count = self.vector_store._collection.count()
            
            # 获取文档元数据
            document_metadata = {}
            try:
                results = self.vector_store._collection.get()
                if results and 'metadatas' in results:
                    for metadata in results['metadatas']:
                        if metadata and 'source' in metadata:
                            source_file = metadata['source']
                            if source_file not in document_metadata:
                                document_metadata[source_file] = {
                                    'chunk_count': 0,
                                    'pages': set()
                                }
                            document_metadata[source_file]['chunk_count'] += 1
                            if 'page' in metadata:
                                document_metadata[source_file]['pages'].add(metadata['page'])
            except Exception as e:
                logger.warning(f"获取文档元数据失败: {e}")
            
            return {
                'total_documents': count,
                'tracked_files': len(document_metadata),
                'document_metadata': document_metadata
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                'total_documents': 0,
                'tracked_files': 0,
                'document_metadata': {}
            }

# 删除重复的 _load_512d_embedder 和 _create_fallback_embedder 方法

SmartVectorStore = VectorStore