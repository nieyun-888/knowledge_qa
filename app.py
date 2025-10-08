# app.py - 修复版只检索模式
import streamlit as st
import os
import logging
from src.vector_store import SmartVectorStore
import requests
import json
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="冰姐问答小课堂",
    page_icon="🎓",
    layout="wide"
)

class DeepSeekAPI:
    """DeepSeek API 管理类 - 腾讯云OpenAI SDK版本"""
    
    def __init__(self):
        self.base_url = "https://api.lkeap.cloud.tencent.com/v1/chat/completions"
        self.model_name = "deepseek-v3.1"
        
        # 初始化session state
        if 'api_key' not in st.session_state:
            st.session_state.api_key = ""
        if 'api_key_set' not in st.session_state:
            st.session_state.api_key_set = False
    
    @property
    def api_key(self):
        return st.session_state.api_key
    
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        if api_key and api_key.strip():
            st.session_state.api_key = api_key.strip()
            st.session_state.api_key_set = True
            st.session_state.api_key_preview = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else api_key
            return True
        else:
            st.error("❌ 请输入有效的API密钥")
            return False
    
    def is_logged_in(self) -> bool:
        """检查是否已登录"""
        return bool(st.session_state.api_key and st.session_state.api_key_set)
    
    def test_api_connection(self) -> Dict:
        """测试API连接"""
        if not self.is_logged_in():
            return {"success": False, "message": "未设置API密钥"}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        test_data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": "请简单回复'连接成功'"
                }
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=test_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                return {
                    "success": True, 
                    "message": f"✅ 腾讯云DeepSeek API连接成功！\n回复: {answer}"
                }
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('error', {}).get('message', '')
                    error_msg = f"{error_msg}: {error_detail}"
                except:
                    pass
                return {"success": False, "message": f"❌ API连接失败: {error_msg}"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"❌ 网络错误: {str(e)}"}
    
    def get_answer(self, question: str, contexts: List[Dict]) -> str:
        """获取DeepSeek答案"""
        if not self.is_logged_in():
            return "请先设置API密钥"
        
        # 构建上下文
        context_text = self._build_context_text(contexts)
        
        prompt = f"""基于以下资料回答问题：

相关资料：
{context_text}

问题：{question}

请根据资料内容给出准确、专业的回答。如果资料中没有相关信息，请如实说明。"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的知识问答助手，根据提供的资料准确回答问题。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ API调用错误：{str(e)}"
    
    def _build_context_text(self, contexts: List[Dict]) -> str:
        """构建上下文文本"""
        context_text = ""
        for i, context in enumerate(contexts, 1):
            source = context.get('source', '未知文档')
            page = context.get('page', '未知页码')
            content = context.get('content', '')
            
            doc_name = os.path.basename(source)
            if '.' in doc_name:
                doc_name = doc_name.rsplit('.', 1)[0]
            
            context_text += f"\n【资料{i}】《{doc_name}》第{page}页：\n{content}\n"
        
        return context_text

@st.cache_resource
def init_vector_store():
    """初始化向量存储 - 只加载现有数据库"""
    try:
        st.info("🔍 正在加载知识库...")
        
        chroma_db_path = "./chroma_db"
        vector_store = SmartVectorStore()
        
        # 检查目录是否存在
        if not os.path.exists(chroma_db_path):
            st.error("❌ 知识库目录不存在")
            return None
        
        # 检查目录内容
        files = os.listdir(chroma_db_path)
        if not files:
            st.error("❌ 知识库目录为空")
            return None
            
        st.info(f"📁 找到知识库，包含 {len(files)} 个文件")
        
        # 直接尝试加载
        if vector_store.load_existing_vector_store():
            st.success("✅ 知识库加载成功！")
            return vector_store
        else:
            st.error("❌ 知识库加载失败")
            # 显示详细错误
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings import HuggingFaceEmbeddings
                
                embedding_model = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-zh-v1.5",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                test_store = Chroma(
                    persist_directory=chroma_db_path,
                    embedding_function=embedding_model,
                    collection_name="pdf_documents"
                )
                count = test_store._collection.count()
                st.info(f"测试加载成功，文档数量: {count}")
            except Exception as e:
                st.error(f"详细错误: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"❌ 初始化失败: {str(e)}")
        return None

def main():
    st.title("🎓 冰姐问答小课堂")
    st.markdown("---")
    
    # 初始化session state
    if 'deepseek_api' not in st.session_state:
        st.session_state.deepseek_api = DeepSeekAPI()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    deepseek_api = st.session_state.deepseek_api
    
    # 侧边栏
    with st.sidebar:
        st.header("🔑 API设置")
        
        api_key = st.text_input(
            "输入腾讯云DeepSeek API密钥:",
            type="password",
            placeholder="sk-...",
            value=st.session_state.get('api_key', '')
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("设置密钥"):
                if deepseek_api.set_api_key(api_key):
                    st.rerun()
        with col2:
            if st.button("测试连接"):
                if deepseek_api.is_logged_in():
                    with st.spinner("测试中..."):
                        result = deepseek_api.test_api_connection()
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
                else:
                    st.error("请先设置API密钥")
        
        st.markdown("---")
        st.info("""
        **使用说明：**
        1. 设置腾讯云DeepSeek API密钥
        2. 测试连接是否成功  
        3. 在下方输入问题开始问答
        """)
        
        # 显示连接状态
        st.markdown("---")
        if st.session_state.api_key_set:
            st.success("✅ API密钥已设置")
        else:
            st.error("❌ API密钥未设置")
    
    # 初始化向量存储
    vector_store = init_vector_store()
    
    # 主界面布局
    if vector_store is None:
        st.error("❌ 知识库未加载，无法进行问答")
        st.info("""
        **请确保:**
        - `chroma_db` 目录已正确上传
        - 向量数据库文件完整
        """)
        # 不显示聊天输入框
        st.stop()
    
    # 知识库加载成功，显示主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 问答对话框")
        st.success("✅ 知识库已加载，可以开始问答")
        
        # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 用户输入 - 只有在知识库加载成功时才显示
        if prompt := st.chat_input("请输入你的问题..."):
            if not st.session_state.api_key_set:
                st.error("❌ 请先在侧边栏设置API密钥")
                st.stop()
            
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 处理回答
            with st.chat_message("assistant"):
                with st.spinner("🔍 搜索知识库..."):
                    try:
                        results = vector_store.search_similar_documents(prompt, k=5)
                        
                        if not results:
                            response = "❌ 未在知识库中找到相关信息。"
                        else:
                            contexts = []
                            for doc in results:
                                contexts.append({
                                    'source': doc.metadata.get('source', '未知'),
                                    'page': doc.metadata.get('page', '未知'),
                                    'content': doc.page_content
                                })
                            
                            # 显示找到的文档片段
                            with st.expander("📚 找到的相关文档片段", expanded=False):
                                for i, context in enumerate(contexts, 1):
                                    doc_name = os.path.basename(context['source'])
                                    if '.' in doc_name:
                                        doc_name = doc_name.rsplit('.', 1)[0]
                                    
                                    st.markdown(f"**片段 {i}**")
                                    st.markdown(f"**文档：** 《{doc_name}》")
                                    st.markdown(f"**页码：** 第{context['page']}页")
                                    st.markdown(f"**内容：** {context['content'][:200]}...")
                                    st.markdown("---")
                            
                            # 获取DeepSeek答案
                            with st.spinner("🤔 生成回答..."):
                                response = deepseek_api.get_answer(prompt, contexts)
                    
                    except Exception as e:
                        response = f"❌ 搜索知识库时出错: {str(e)}"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.subheader("📊 知识库状态")
        try:
            stats = vector_store.get_document_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总文档数", stats.get('total_documents', 'N/A'))
            with col2:
                st.metric("跟踪文件数", stats.get('tracked_files', 'N/A'))
            
            # 显示文档列表
            with st.expander("📋 文档列表"):
                metadata = stats.get('document_metadata', {})
                if metadata:
                    for file_name in list(metadata.keys())[:10]:
                        doc_name = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                        st.write(f"• 《{doc_name}》")
                else:
                    st.info("暂无文档信息")
                    
        except Exception as e:
            st.error(f"获取统计信息失败: {str(e)}")
        
        st.markdown("---")
        if st.button("清空聊天记录"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
