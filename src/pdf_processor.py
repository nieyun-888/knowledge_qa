import os
import logging
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """加载单个PDF文件"""
        try:
            logger.info(f"正在加载PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata["source"] = os.path.basename(pdf_path)
                doc.metadata["file_path"] = pdf_path
            
            logger.info(f"成功加载 {len(documents)} 页内容来自 {pdf_path}")
            return documents
        except Exception as e:
            logger.error(f"加载PDF失败 {pdf_path}: {str(e)}")
            return []
    
    def load_pdfs_from_directory(self, pdf_dir: str) -> List[Document]:
        """从目录加载所有PDF文件"""
        all_documents = []
        
        if not os.path.exists(pdf_dir):
            logger.error(f"PDF目录不存在: {pdf_dir}")
            return all_documents
        
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"在目录 {pdf_dir} 中未找到PDF文件")
            return all_documents
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            documents = self.load_pdf(pdf_path)
            all_documents.extend(documents)
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为小块"""
        if not documents:
            logger.warning("没有文档可供分割")
            return []
        
        logger.info(f"开始分割 {len(documents)} 个文档")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"分割完成，共生成 {len(chunks)} 个文本块")
        
        return chunks