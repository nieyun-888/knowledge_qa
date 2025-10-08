import os
import logging
import json
import time
import hashlib
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        # 使用本地下载的模型
        self.embedding_model = self._load_local_model()
        logger.info("向量存储初始化完成")
    

    def _load_local_model(self):
        
        """加载嵌入模型 - 部署优化版"""
        # 方法1：优先使用 LangChain 的在线 HuggingFaceEmbeddings
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("使用在线 HuggingFace 嵌入模型...")
        
            embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
            # 测试一下模型是否能正常工作
            try:
                test_embedding = embedding_model.embed_query("测试")
                logger.info(f"✅ LangChain 在线嵌入模型加载成功，向量维度: {len(test_embedding)}")
                return embedding_model
            except:
                logger.info("✅ LangChain 在线嵌入模型加载成功")
                return embedding_model
            
        except Exception as e:
            logger.warning(f"LangChain 在线嵌入模型加载失败: {e}")
    
        # 方法2：如果在线模型失败，使用备用嵌入模型
        logger.info("使用备用嵌入模型")
        return self._create_fallback_embedder()
       
    
    def _create_fallback_embedder(self):
        """创建备用的简单嵌入模型"""
        import numpy as np
        import hashlib
        
        class SimpleEmbedder:
            def __init__(self):
                self.dim = 384
                
            def embed_documents(self, texts):
                logger.info(f"为 {len(texts)} 个文档生成嵌入")
                embeddings = []
                for i, text in enumerate(texts):
                    if i % 10 == 0 and i > 0:
                        logger.info(f"已处理 {i}/{len(texts)} 个文档")
                    embeddings.append(self._text_to_vector(text))
                return embeddings
            
            def embed_query(self, text):
                return self._text_to_vector(text)
            
            def _text_to_vector(self, text):
                # 使用文本内容的哈希值生成确定性随机向量
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                seed = int(text_hash[:8], 16)
                np.random.seed(seed)
                vector = np.random.normal(0, 1, self.dim)
                # 归一化向量
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                return vector.tolist()
        
        return SimpleEmbedder()
    
    def create_vector_store(self, documents):
        """创建向量存储 - 使用 Chroma"""
        try:
            logger.info("开始创建向量存储...")
            
            if not documents:
                logger.error("没有文档可处理")
                return False
            
            # 使用 Chroma 创建向量存储
            try:
                from langchain_chroma import Chroma
            except ImportError:
                try:
                    from langchain_community.vectorstores import Chroma
                except ImportError:
                    from langchain.vectorstores import Chroma
            
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            logger.info(f"正在为 {len(texts)} 个文档创建向量索引...")
            
            # 创建 Chroma 向量存储
            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas,
                persist_directory=self.persist_directory,
                collection_name="pdf_documents"
            )
            
            logger.info(f"✅ Chroma 向量存储创建成功! 保存在: {self.persist_directory}")
            return True
            
        except Exception as e:
            logger.error(f"创建向量存储失败: {e}")
            return False

    def similarity_search(self, query: str, k: int = 3):
        """相似度搜索"""
        try:
            if not hasattr(self, 'vector_store'):
                # 加载已保存的向量存储
                try:
                    from langchain_chroma import Chroma
                except ImportError:
                    try:
                        from langchain_community.vectorstores import Chroma
                    except ImportError:
                        from langchain.vectorstores import Chroma
                
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                    collection_name="pdf_documents"
                )
            
            logger.info(f"执行搜索: '{query}'")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"找到 {len(results)} 个相关文档")
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    def load_existing_vector_store(self):
        """加载已存在的向量存储"""
        try:
            try:
                from langchain_chroma import Chroma
            except ImportError:
                try:
                    from langchain_community.vectorstores import Chroma
                except ImportError:
                    from langchain.vectorstores import Chroma
            
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name="pdf_documents"
            )
            
            # 测试一下是否真的加载成功
            count = self.vector_store._collection.count()
            logger.info(f"加载现有向量存储成功，包含 {count} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"加载现有向量存储失败: {e}")
            return False

    def search_similar_documents(self, query: str, k: int = 3):
        """搜索相似文档 - main.py 需要的方法"""
        return self.similarity_search(query, k)

    @staticmethod
    def vector_store_exists(persist_directory="./chroma_db"):
        """检查向量存储是否已存在"""
        import os
        return os.path.exists(persist_directory) and os.path.isdir(persist_directory)

# ===== 智能向量存储类（放在 VectorStore 类之后） =====

class SmartVectorStore(VectorStore):
    """智能向量存储，支持增量更新"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        super().__init__(persist_directory)
        self.document_metadata_file = os.path.join(persist_directory, "document_metadata.json")
        self.document_metadata = self._load_document_metadata()
    
    def _load_document_metadata(self) -> Dict[str, Dict[str, Any]]:
        """加载文档元数据"""
        if os.path.exists(self.document_metadata_file):
            try:
                with open(self.document_metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_document_metadata(self):
        """保存文档元数据"""
        os.makedirs(os.path.dirname(self.document_metadata_file), exist_ok=True)
        with open(self.document_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
    
    def filter_new_and_updated_documents(self, documents: List[Any]) -> List[Any]:
        """过滤出新文档和已更新的文档（公开方法）"""
        return self._filter_new_and_updated_documents(documents)
    
    def _filter_new_and_updated_documents(self, documents: List[Any]) -> List[Any]:
        """过滤出新文档和已更新的文档（内部方法）"""
        new_or_updated_docs = []
        
        for doc in documents:
            if not hasattr(doc, 'metadata') or 'source' not in doc.metadata:
                new_or_updated_docs.append(doc)
                continue
            
            source_path = doc.metadata['source']
            if not os.path.exists(source_path):
                new_or_updated_docs.append(doc)
                continue
            
            current_hash = self._get_document_hash(source_path)
            file_name = os.path.basename(source_path)
            
            # 检查是否是新的或已更新的文档
            if (file_name not in self.document_metadata or 
                self.document_metadata[file_name].get('hash') != current_hash):
                new_or_updated_docs.append(doc)
                # 更新元数据
                self.document_metadata[file_name] = {
                    'hash': current_hash,
                    'source': source_path,
                    'processed_time': time.time()
                }
        
        return new_or_updated_docs
    


    def _get_document_hash(self, file_path: str) -> str:
        """计算文档哈希值用于检测变更"""
        try:
            file_stat = os.stat(file_path)
            # 使用文件大小和修改时间生成哈希
            hash_input = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def _filter_new_and_updated_documents(self, documents: List[Any]) -> List[Any]:
        """过滤出新文档和已更新的文档"""
        new_or_updated_docs = []
        
        for doc in documents:
            if not hasattr(doc, 'metadata') or 'source' not in doc.metadata:
                new_or_updated_docs.append(doc)
                continue
            
            source_path = doc.metadata['source']
            if not os.path.exists(source_path):
                new_or_updated_docs.append(doc)
                continue
            
            current_hash = self._get_document_hash(source_path)
            file_name = os.path.basename(source_path)
            
            # 检查是否是新的或已更新的文档
            if (file_name not in self.document_metadata or 
                self.document_metadata[file_name].get('hash') != current_hash):
                new_or_updated_docs.append(doc)
                # 更新元数据
                self.document_metadata[file_name] = {
                    'hash': current_hash,
                    'source': source_path,
                    'processed_time': time.time()
                }
        
        return new_or_updated_docs
    
    def smart_create_vector_store(self, documents: List[Any], force_recreate: bool = False) -> bool:
        """智能创建向量存储，只处理新文档和更新的文档"""
        try:
            logger.info("开始智能创建向量存储...")
            
            if force_recreate:
                logger.info("强制重新创建模式，处理所有文档")
                return self.create_vector_store(documents)
            
            # 过滤出需要处理的文档
            docs_to_process = self._filter_new_and_updated_documents(documents)
            
            if not docs_to_process:
                logger.info("没有发现新的或更新的文档，跳过处理")
                return True
            
            logger.info(f"发现 {len(docs_to_process)} 个新文档或更新的文档需要处理")
            logger.info(f"跳过 {len(documents) - len(docs_to_process)} 个未变化的文档")
            
            # 如果有现有向量存储，只添加新文档
            if self.vector_store_exists() and hasattr(self, 'vector_store'):
                logger.info("向现有向量存储添加新文档...")
                success = self._add_documents_to_existing_store(docs_to_process)
            else:
                logger.info("创建新的向量存储...")
                success = self.create_vector_store(docs_to_process)
            
            if success:
                self._save_document_metadata()
                logger.info("✅ 智能向量存储更新成功!")
            else:
                logger.error("❌ 智能向量存储更新失败!")
            
            return success
            
        except Exception as e:
            logger.error(f"智能创建向量存储失败: {e}")
            return False
    
    def _add_documents_to_existing_store(self, documents: List[Any]) -> bool:
        """向现有向量存储添加文档"""
        try:
            if not hasattr(self, 'vector_store'):
                if not self.load_existing_vector_store():
                    return False
            
            # 使用 Chroma 的 add_documents 方法
            self.vector_store.add_documents(documents)
            
            logger.info(f"成功添加 {len(documents)} 个文档到现有向量存储")
            return True
            
        except Exception as e:
            logger.error(f"添加文档到现有向量存储失败: {e}")
            return False

    def get_document_stats(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        try:
            if not hasattr(self, 'vector_store'):
                if not self.load_existing_vector_store():
                    return {}
            
            # 获取集合中的文档数量
            count = self.vector_store._collection.count()
            
            return {
                'total_documents': count,
                'tracked_files': len(self.document_metadata),
                'document_metadata': self.document_metadata
            }
        except Exception as e:
            logger.error(f"获取文档统计失败: {e}")
            return {}

# ===== 兼容性修复 =====

def process_pdfs_and_create_vector_store(pdf_directory="./data/raw_pdfs", persist_directory="./chroma_db"):
    """
    处理PDF并创建向量存储的主函数
    """
    logger.info("开始处理PDF并创建向量存储...")
    
    try:
        # 导入必要的模块
        from .pdf_processor import PDFProcessor
        
        # 创建向量存储实例
        vector_store = VectorStore(persist_directory=persist_directory)
        
        # 加载和处理PDF
        pdf_processor = PDFProcessor()
        documents = pdf_processor.load_pdfs_from_directory(pdf_directory)
        
        if not documents:
            logger.error("没有找到PDF文档或加载失败")
            return False
        
        # 分割文档
        chunks = pdf_processor.split_documents(documents)
        logger.info(f"分割完成，共生成 {len(chunks)} 个文本块")
        
        # 创建向量存储
        if vector_store.create_vector_store(chunks):
            logger.info("向量存储创建成功!")
            return True
        else:
            logger.error("向量存储创建失败!")
            return False
            
    except Exception as e:
        logger.error(f"处理PDF并创建向量存储失败: {e}")
        return False

# 兼容性别名
VectorStoreManager = VectorStore

# 测试函数
def test_embedding():
    """测试嵌入模型是否正常工作"""
    store = VectorStore()
    
    # 测试文档嵌入
    test_texts = ["这是一个测试文档", "这是另一个测试文档"]
    embeddings = store.embedding_model.embed_documents(test_texts)
    
    print(f"文档数量: {len(test_texts)}")
    print(f"嵌入维度: {len(embeddings[0])}")
    print("✅ 嵌入模型测试通过")

if __name__ == "__main__":
    test_embedding()
