# app.py - 只检索模式版本
import streamlit as st
import os
import logging
from src.vector_store import SmartVectorStore
import requests
import json
from typing import List, Dict, Any
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="冰姐问答小课堂",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DeepSeekAPI:
    """DeepSeek API 管理类 - 腾讯云OpenAI SDK版本"""
    
    def __init__(self):
        # 腾讯云OpenAI SDK接入端点
        self.base_url = "https://api.lkeap.cloud.tencent.com/v1/chat/completions"
        # 使用session_state来持久化API密钥
        if 'api_key' not in st.session_state:
            st.session_state.api_key = None
        if 'api_key_set' not in st.session_state:
            st.session_state.api_key_set = False
        # 你的腾讯云OpenAI SDK API密钥
        # 添加模型名称 - 使用腾讯云支持的模型
        self.model_name = "deepseek-v3.1"
    
    @property
    def api_key(self):
        """获取API密钥"""
        return st.session_state.api_key
    
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        st.session_state.api_key = api_key.strip()
        st.session_state.api_key_set = True
        st.session_state.api_key_preview = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else api_key
    
    def login_with_password(self, password: str) -> bool:
        """密码登录方式"""
        st.error("密码登录功能在线上版本不可用，请使用API直接登录")
        return False
    
    def is_logged_in(self) -> bool:
        """检查是否已登录"""
        return st.session_state.api_key is not None and st.session_state.api_key.strip() != ""
    
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
                    "content": "请简单回复'腾讯云API连接成功'"
                }
            ],
            "max_tokens": 20,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=test_data, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                return {
                    "success": True, 
                    "message": f"✅ 腾讯云DeepSeek API连接成功！\n回复: {answer}",
                    "response": result
                }
            else:
                error_detail = "未知错误"
                try:
                    error_json = response.json()
                    error_detail = error_json.get('error', {}).get('message', response.text)
                except:
                    error_detail = response.text[:200]
                
                return {
                    "success": False, 
                    "message": f"❌ API返回错误: HTTP {response.status_code}",
                    "detail": error_detail
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False, 
                "message": f"❌ 网络连接错误: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False, 
                "message": f"❌ 未知错误: {str(e)}"
            }
    
    def get_answer(self, question: str, contexts: List[Dict]) -> str:
        """获取DeepSeek答案"""
        if not self.is_logged_in():
            return "请先登录DeepSeek API"
        
        # 构建上下文提示
        context_text = self._build_context_text(contexts)
        
        prompt = f"""你是一个专业的知识问答助手。请根据提供的上下文信息回答用户问题。

上下文信息来自多个文档资料：
{context_text}

用户问题：{question}

请按照以下格式回答：

## 🔍 思考过程

我主要从以下几个资料中找到了相关信息：

### 1. 第一个资料来源
- **文档名称**：《文档名称》
- **页码**：第X页
- **相关内容**：从该文档中找到的具体内容描述...
- **关键信息**：提取的关键知识点...

### 2. 第二个资料来源  
- **文档名称**：《文档名称》
- **页码**：第X页
- **相关内容**：从该文档中找到的具体内容描述...
- **关键信息**：提取的关键知识点...

### 3. 第三个资料来源
- **文档名称**：《文档名称》
- **页码**：第X页  
- **相关内容**：从该文档中找到的具体内容描述...
- **关键信息**：提取的关键知识点...

### 4. 第四个资料来源
- **文档名称**：《文档名称》
- **页码**：第X页  
- **相关内容**：从该文档中找到的具体内容描述...
- **关键信息**：提取的关键知识点...

### 5. 第五个资料来源
- **文档名称**：《文档名称》
- **页码**：第X页  
- **相关内容**：从该文档中找到的具体内容描述...
- **关键信息**：提取的关键知识点...

## 💡 综合答案

基于以上分析，我的答案是：
[这里给出简洁明确的答案]

## 📚 推荐阅读

如果同学想深入了解，建议查阅：
- 《文档名称》第X页：[具体章节或内容]
- 《文档名称》第X页：[具体章节或内容]
- 《文档名称》第X页：[具体章节或内容]
- 《文档名称》第X页：[具体章节或内容]
- 《文档名称》第X页：[具体章节或内容]

请确保准确引用文档名称、页码和具体内容，答案要专业、准确、简洁。"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一位专业的法律教育助手，擅长从多个文档资料中提取准确信息并给出结构清晰的回答。你的回答要严谨、专业、易于理解。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"❌ API调用错误：{str(e)}"
        except Exception as e:
            return f"❌ 处理错误：{str(e)}"
    
    def _build_context_text(self, contexts: List[Dict]) -> str:
        """构建上下文文本"""
        context_text = ""
        for i, context in enumerate(contexts, 1):
            source = context.get('source', '未知文档')
            page = context.get('page', '未知页码')
            content = context.get('content', '')
            
            # 提取文档名称（去掉路径和扩展名）
            doc_name = os.path.basename(source)
            if '.' in doc_name:
                doc_name = doc_name.rsplit('.', 1)[0]
            
            context_text += f"\n【资料{i}】\n"
            context_text += f"文档名称：《{doc_name}》\n"
            context_text += f"页码：第{page}页\n"
            context_text += f"内容片段：{content}\n"
            context_text += "-" * 60 + "\n"
        
        return context_text

@st.cache_resource
def init_vector_store():
    """初始化向量存储 - 只加载现有数据库，不处理新文档"""
    try:
        st.info("🔍 只检索模式：正在加载现有向量数据库...")
        
        # 检查向量数据库目录是否存在
        chroma_db_path = "./chroma_db"
        if not os.path.exists(chroma_db_path):
            st.error(f"❌ 向量数据库目录不存在: {chroma_db_path}")
            return None
            
        # 检查目录内容
        files = os.listdir(chroma_db_path)
        st.info(f"📁 找到向量数据库目录，包含 {len(files)} 个文件")
        st.info(f"📄 文件列表: {files}")
        
        # 创建向量存储实例
        vector_store = SmartVectorStore()
        
        # 检查向量存储是否存在
        if vector_store.vector_store_exists():
            st.info("🔄 正在加载现有向量存储...")
            if vector_store.load_existing_vector_store():
                st.success("✅ 知识库加载成功！准备就绪")
                return vector_store
            else:
                st.error("❌ 向量存储加载失败")
                # 添加详细错误信息
                try:
                    # 尝试直接加载查看具体错误
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
        else:
            st.error("❌ 未找到可用的向量存储")
            return None
            
    except Exception as e:
        st.error(f"❌ 初始化失败: {str(e)}")
        return None

def main():

    # 初始化
    # 主标题
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🎓 冰姐问答小课堂 - 只检索模式</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 初始化session state
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 使用session_state来持久化DeepSeekAPI实例
    if 'deepseek_api' not in st.session_state:
        st.session_state.deepseek_api = DeepSeekAPI()
    
    # 在这里使用session_state中的deepseek_api实例
    deepseek_api = st.session_state.deepseek_api




    # 在这里创建 DeepSeekAPI 实例，确保在整个函数中可用
    #deepseek_api = DeepSeekAPI()  # 新增这行
    
    # 侧边栏 - API设置
    with st.sidebar:
        st.header("🔑 API设置 - 只检索模式")
        
        # 删除这里的 deepseek_api = DeepSeekAPI() 行
        # 显示当前使用的API端点
        st.info(f"**API端点:**\n`{deepseek_api.base_url}`")
        
        # API登录方式选择
        login_method = st.radio("选择登录方式:", ["直接输入API", "密码登录"])
        
        if login_method == "直接输入API":
            api_key = st.text_input("输入腾讯云DeepSeek API密钥:", type="password", placeholder="sk-...")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("设置API密钥"):
                    if api_key:
                        deepseek_api.set_api_key(api_key)
                        st.success("✅ API密钥已设置")
                        st.rerun()
                    else:
                        st.error("❌ 请输入有效的API密钥")
            with col2:
                if st.button("测试连接"):
                    if deepseek_api.is_logged_in():
                        with st.spinner("测试连接中..."):
                            result = deepseek_api.test_api_connection()
                        if result["success"]:
                            st.success(result["message"])
                        else:
                            st.error(f"{result['message']}")
                            if "detail" in result:
                                st.error(f"详情: {result['detail']}")
                    else:
                        st.error("请先设置API密钥")
            
        else:  # 密码登录
            password = st.text_input("输入密码:", type="password", placeholder="输入密码'请输入密码'")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("登录"):
                    if deepseek_api.login_with_password(password):
                        st.success("✅ 登录成功！API密钥已自动填充")
                        st.rerun()
                    else:
                        st.error("❌ 密码错误")
            with col2:
                if st.button("测试连接"):
                    if deepseek_api.is_logged_in():
                        with st.spinner("测试连接中..."):
                            result = deepseek_api.test_api_connection()
                        if result["success"]:
                            st.success(result["message"])
                        else:
                            st.error(f"{result['message']}")
                            if "detail" in result:
                                st.error(f"详情: {result['detail']}")
                    else:
                        st.error("请先登录")
        
        st.markdown("---")
        st.markdown("### 📋 系统模式")
        st.warning("**当前模式: 只检索模式**")
        st.info("""
        **功能说明:**
        - ✅ 加载现有知识库
        - ✅ 智能问答检索
        - ❌ 不处理新PDF文档
        - ❌ 不重新创建向量库
        
        **数据处理请运行:**
        ```bash
        python main.py
        ```
        选择模式1、2或4
        """)
        
        # 显示连接状态
        st.markdown("---")
        st.subheader("连接状态")
        if st.session_state.api_key_set:
            st.success("✅ API密钥已设置")
            if deepseek_api.is_logged_in():
                st.info(f"密钥: {st.session_state.get('api_key_preview', '已设置')}")
        else:
            st.error("❌ API密钥未设置")
    
    # 初始化向量存储 - 只检索模式
    vector_store = init_vector_store()
    
    # 主界面 - 聊天区域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 问答对话框 - 只检索模式")
        
        # 显示模式说明
        if vector_store is None:
            st.error("❌ 知识库未加载，无法进行问答")
            # 将提示从"运行main.py"改为：
            st.info("""
            **请确保:**
            - `chroma_db` 目录已正确上传
            - 向量数据库文件完整
            """)
        else:
            st.success("✅ 知识库已加载，可以开始问答")
        
        # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 用户输入
        if prompt := st.chat_input("请输入你的问题..."):
            # 检查API登录状态
            if not st.session_state.api_key_set:
                st.error("❌ 请先在侧边栏设置DeepSeek API")
                st.stop()
            
            # 检查向量存储
            if vector_store is None:
                st.error("❌ 知识库未加载，请先运行 main.py 处理PDF文档")
                st.stop()
            
            # 添加用户消息到聊天历史
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 处理用户问题
            with st.chat_message("assistant"):
                with st.spinner("🔍 正在搜索知识库..."):
                    # 搜索相关文档 - 只检索模式
                    try:
                        results = vector_store.search_similar_documents(prompt, k=5)
                        st.info(f"🔍 从知识库中找到 {len(results)} 个相关文档片段")
                        
                        if not results:
                            response = "❌ 未在知识库中找到相关信息，请尝试其他问题。"
                        else:
                            # 构建上下文信息
                            contexts = []
                            for i, doc in enumerate(results):
                                source = doc.metadata.get('source', '未知文档')
                                page = doc.metadata.get('page', '未知页码')
                                content = doc.page_content
                                
                                contexts.append({
                                    'source': source,
                                    'page': page,
                                    'content': content
                                })
                            
                            # 显示找到的文档片段
                            with st.expander("📚 找到的相关文档片段（点击查看）", expanded=False):
                                for i, context in enumerate(contexts, 1):
                                    doc_name = os.path.basename(context['source'])
                                    if '.' in doc_name:
                                        doc_name = doc_name.rsplit('.', 1)[0]
                                    
                                    st.markdown(f"**📄 片段 {i}**")
                                    st.markdown(f"**文档：** 《{doc_name}》")
                                    st.markdown(f"**页码：** 第{context['page']}页")
                                    st.markdown(f"**内容预览：** {context['content'][:200]}...")
                                    st.markdown("---")
                            
                            # 获取DeepSeek答案
                            with st.spinner("🤔 正在分析文档并生成答案..."):
                                response = deepseek_api.get_answer(prompt, contexts)
                    
                    except Exception as e:
                        response = f"❌ 搜索知识库时出错: {str(e)}"
                
                # 显示助手回答
                st.markdown(response)
                
                # 添加到聊天历史
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.subheader("📊 知识库状态")
        if vector_store:
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
                        for file_name, info in list(metadata.items())[:10]:
                            doc_name = file_name
                            if '.' in doc_name:
                                doc_name = doc_name.rsplit('.', 1)[0]
                            st.write(f"• 《{doc_name}》")
                    else:
                        st.info("暂无文档信息")
                        
                # 操作说明
                st.markdown("---")
                st.info("""
                **只检索模式说明:**
                - 仅使用现有知识库
                - 不处理新PDF文档
                - 数据处理请运行main.py
                """)
                        
            except Exception as e:
                st.error(f"获取统计信息失败: {str(e)}")
        else:
            st.error("❌ 知识库未加载")
            st.info("请运行 `python main.py` 先处理PDF文档")
        
        st.markdown("---")
        st.subheader("🛠️ 操作")
        if st.button("清空聊天记录"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":

    main()
