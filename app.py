import streamlit as st
import os
import requests
from typing import List
from src.vector_store import SmartVectorStore

st.set_page_config(page_title="知识问答", layout="wide")

class DeepSeekAPI:
    def __init__(self):
        self.base_url = "https://api.lkeap.cloud.tencent.com/v1/chat/completions"
        self.model_name = "deepseek-v3.1"
        
    def set_api_key(self, api_key: str):
        if api_key and api_key.strip():
            st.session_state.api_key = api_key.strip()
            st.session_state.api_key_set = True
            return True
        return False
    
    def is_logged_in(self):
        return hasattr(st.session_state, 'api_key_set') and st.session_state.api_key_set
    
    def get_answer(self, question: str, contexts: List[dict]) -> str:
        if not self.is_logged_in():
            return "请先设置API密钥"
        
        # 构建上下文
        context_text = "\n".join([f"【资料{i}】{ctx['content']}" 
                                for i, ctx in enumerate(contexts, 1)])
        
        prompt = f"""基于以下资料回答问题：

{context_text}

问题：{question}

请根据资料给出准确回答。"""

        headers = {
            "Authorization": f"Bearer {st.session_state.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"API错误：{str(e)}"

@st.cache_resource
def init_vector_store():
    """初始化向量存储"""
    if not os.path.exists("./chroma_db"):
        st.error("❌ 知识库不存在")
        return None
    
    vector_store = SmartVectorStore()
    if vector_store.load_existing_vector_store():
        st.success("✅ 知识库加载成功")
        return vector_store
    else:
        st.error("❌ 知识库加载失败")
        return None

def main():
    st.title("🎓 知识问答系统")
    
    # 初始化
    if 'deepseek_api' not in st.session_state:
        st.session_state.deepseek_api = DeepSeekAPI()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 侧边栏
    with st.sidebar:
        st.header("🔑 API设置")
        api_key = st.text_input("API密钥:", type="password")
        if st.button("设置密钥") and api_key:
            if st.session_state.deepseek_api.set_api_key(api_key):
                st.success("✅ 密钥已设置")
    
    # 加载知识库
    vector_store = init_vector_store()
    
    if not vector_store:
        st.stop()
    
    # 主界面
    st.subheader("💬 问答")
    
    # 显示历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 用户输入
    if prompt := st.chat_input("输入问题..."):
        if not st.session_state.deepseek_api.is_logged_in():
            st.error("请先设置API密钥")
            st.stop()
        
        # 用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 助手回答
        with st.chat_message("assistant"):
            with st.spinner("搜索中..."):
                results = vector_store.search_similar_documents(prompt, k=3)
                
                if not results:
                    response = "未找到相关信息"
                else:
                    contexts = [{
                        'content': doc.page_content,
                        'source': doc.metadata.get('source', '未知')
                    } for doc in results]
                    
                    response = st.session_state.deepseek_api.get_answer(prompt, contexts)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
