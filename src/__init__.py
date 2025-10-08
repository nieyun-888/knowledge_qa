"""
knowledge_qa 核心模块
向量存储功能（只检索模式）
"""

# 修复导入问题
try:
    from .vector_store import VectorStoreManager
except ImportError:
    # 如果 VectorStoreManager 不存在，使用 VectorStore 作为替代
    from .vector_store import VectorStore as VectorStoreManager

# 版本信息
__version__ = "0.1.0"
__author__ = "Your Name"

# 定义公开的API接口（移除 PDFProcessor）
__all__ = [
    "VectorStoreManager",
]

# 包级别的配置
DEFAULT_SETTINGS = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "BAAI/bge-small-zh-v1.5"
}

# 初始化日志配置（可选）
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

print(f"加载 knowledge_qa.src 模块 v{__version__} (只检索模式)")
