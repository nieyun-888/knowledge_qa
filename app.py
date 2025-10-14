import streamlit as st
import os
import requests
from typing import List
from src.vector_store import SmartVectorStore
from math import floor
import logging





st.set_page_config(page_title="冰姐问答小课堂", layout="wide")

from src.image_processor import image_processor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DeepSeekAPI:
    def __init__(self):
        self.base_url = "https://api.lkeap.cloud.tencent.com/v1/chat/completions"
        self.model_name = "deepseek-v3.1"
    
    def set_api_key(self, api_key: str):
        if api_key and api_key.strip():
            st.session_state.api_key = api_key.strip()
            self.api_key = api_key.strip()
            st.session_state.api_key_set = True
            st.success("✅ API密钥已设置")
            return True
        st.error("❌ 请输入有效的API密钥")
        return False
    
    def is_logged_in(self):
        return hasattr(st.session_state, 'api_key_set') and st.session_state.api_key_set
    
    def test_connection(self):
        """测试API连接"""
        if not self.is_logged_in():
            return "请先设置API密钥"
        
        headers = {
            "Authorization": f"Bearer {st.session_state.api_key}",
            "Content-Type": "application/json"
        }
        
        test_data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "请回复'测试成功'"}],
            "max_tokens": 10
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=test_data, timeout=30)
            if response.status_code == 200:
                return f"✅ 连接成功: {response.json()['choices'][0]['message']['content']}"
            else:
                return f"❌ 连接失败: HTTP {response.status_code}"
        except Exception as e:
            return f"❌ 连接错误: {str(e)}"
    
    def get_answer_with_new_template(self, question: str, contexts: List[dict], conversation_history: List[dict] = None) -> str:
        """使用新的冰姐风格模板获取答案，包含对话历史"""
        if not self.is_logged_in():
            return "请先设置API密钥"
        
        # 构建上下文文本
        context_text = self._build_context_text(contexts)
        
        # 构建对话历史文本
        history_text = ""
        if conversation_history:
            history_text = "\n\n之前的对话历史：\n"
            for msg in conversation_history[-6:]:  # 只保留最近6条消息，避免太长
                role = "用户" if msg["role"] == "user" else "冰姐"
                history_text += f"{role}：{msg['content']}\n"
        
        # 新的冰姐风格提示词模板
        prompt = f"""你是一个亲切的法律教育助手"冰姐"，请根据提供的上下文信息、对话历史和当前问题，用温暖亲切的语气回答用户问题。

上下文信息来自多个文档资料：
{context_text}

{history_text}

请特别注意：当前用户的问题"{question}"可能是基于之前对话的延续。请仔细理解对话历史，确保回答与之前的对话内容连贯一致。

如果用户的问题涉及之前讨论的例子、概念或内容，请基于之前的对话继续解释，保持话题的连贯性。

请按照以下亲切的风格和格式回答：

乖，冰姐刚才看了一下你的问题。冰姐先对你的问题给你一个直接的答复：
[这里给出直接简洁的答案，特别注意与之前对话的连贯性]

为什么会有这样的结论呢，冰姐主要是从以下几个资料来思考的。

1. 从《文档名称》的第x页，相关内容为：[具体内容]。从这个资料，我们可以得到[关键知识点]...
2. 从《文档名称》的第x页，相关内容为：[具体内容]。从这个资料，我们可以得到[关键知识点]...
3. 从《文档名称》的第x页，相关内容为：[具体内容]。从这个资料，我们可以得到[关键知识点]...
4. 从《文档名称》的第x页，相关内容为：[具体内容]。从这个资料，我们可以得到[关键知识点]...
5. 从《文档名称》的第x页，相关内容为：[具体内容]。从这个资料，我们可以得到[关键知识点]...

因此，冰姐结合以上几个资料，综合得到了这个结论。不知道你理解了没有呢？冰姐再给你举一个相关的例子帮助你理解吧。例：[举一个与当前问题和之前对话都相关的具体生动例子]

这下你应该差不多可以理解了吧。如果眼神还是清澈的话，就看一下以下几个资料再巩固理解一下。

1. 《文档名称》第x页，[具体章节或内容位置]
2. 《文档名称》第x页，[具体章节或内容位置]  
3. 《文档名称》第x页，[具体章节或内容位置]
4. 《文档名称》第x页，[具体章节或内容位置]
5. 《文档名称》第x页，[具体章节或内容位置]

请确保：
- 里面的页码我只是示例，你要从搜索到的数据库中提取出页码，并进行替换
- 使用温暖亲切的"冰姐"语气
- 准确引用文档名称、页码和具体内容
- 例子要生动易懂且与对话历史相关
- 答案要专业、准确、亲切
- 特别注意保持对话的连贯性，理解用户问题中的指代关系
- 如果用户的问题与之前的对话相关，请结合对话历史来理解问题的上下文"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": "你是一位亲切的法律教育助手'冰姐'，擅长用温暖亲切的语气从多个文档资料中提取准确信息，并用生动易懂的方式给出回答。你特别注重对话的连贯性，能够理解用户问题中的指代关系（如'这个例子'、'刚才说的'等），并基于之前的对话上下文给出连贯的回答。你的回答要专业、准确、亲切、温暖。"
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
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"API错误：{str(e)}"


    def _build_context_text(self, contexts: List[dict]) -> str:
        """构建上下文文本 - 从原始代码复制"""
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
    """初始化向量存储"""
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
            # 获取文档数量并显示
            try:
                count = vector_store.vector_store._collection.count()
                st.success(f"✅ 知识库加载成功！共 {count} 个文本块")
                
                # 显示文档统计信息
                with st.expander("📊 知识库详情", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总文本块数", count)
                    with col2:
                        # 估算文档数量（假设每个PDF生成约50-100个文本块）
                        estimated_docs = max(1, count // 80)
                        st.metric("估算PDF数量", f"~{estimated_docs}")
                    with col3:
                        st.metric("向量维度", "自动检测")
                        
            except Exception as e:
                st.success("✅ 知识库加载成功！")
                st.warning(f"⚠️ 无法获取详细统计: {str(e)}")
                
            return vector_store
        else:
            st.error("❌ 知识库加载失败")
            return None
            
    except Exception as e:
        st.error(f"❌ 初始化失败: {str(e)}")
        return None


def main():
    st.title("🎓 冰姐问答小课堂")

    # 🔴 在这里添加：快速搜索测试
    if 'quick_test_done' not in st.session_state:
        st.session_state.quick_test_done = True
        st.info("🧪 正在初始化系统...")
        
        # 初始化向量存储
        vector_store = init_vector_store()
        if vector_store:
            try:
                # 测试搜索功能
                test_query = "法律"
                with st.spinner("验证搜索功能..."):
                    test_results = vector_store.search_similar_documents(test_query, k=1)
                
                if test_results:
                    st.success(f"✅ 系统初始化成功！搜索验证成功，找到 {len(test_results)} 个相关文档")
                    # 显示测试结果预览
                    with st.expander("查看验证结果详情", expanded=False):
                        st.write(f"示例查询: '{test_query}'")
                        st.write(f"相关文档: {test_results[0].metadata.get('source', '未知')}")
                        st.write(f"内容预览: {test_results[0].page_content[:100]}...")
                else:
                    st.warning("⚠️ 搜索验证返回空结果，知识库可能不包含相关示例内容")
            except Exception as e:
                st.error(f"❌ 系统初始化失败: {str(e)}")
        else:
            st.error("❌ 系统初始化失败：向量库加载失败")
        
        st.markdown("---")  # 添加分隔线



    # 初始化
    if 'deepseek_api' not in st.session_state:
        st.session_state.deepseek_api = DeepSeekAPI()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    # 添加图片相关的session_state初始化
    if 'image_question' not in st.session_state:
        st.session_state.image_question = ""
    if 'image_processed' not in st.session_state:
        st.session_state.image_processed = False
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    if 'ocr_result' not in st.session_state:
        st.session_state.ocr_result = None
    
    # 侧边栏
    with st.sidebar:
        st.header("🔑 API设置")
        api_key = st.text_input("输入腾讯云DeepSeek API密钥:", type="password", placeholder="sk-...")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("设置密钥"):
                st.session_state.deepseek_api.set_api_key(api_key)
        with col2:
            if st.button("检查连接"):
                if st.session_state.deepseek_api.is_logged_in():
                    result = st.session_state.deepseek_api.test_connection()
                    st.info(result)
                else:
                    st.error("请先设置API密钥")
    
    # 加载知识库
    vector_store = init_vector_store()
    
    # 主界面
     # 主界面
    st.subheader("💬 冰姐问答小课堂")
    if not vector_store:
        st.error("❌ 知识库加载失败，无法进行问答")
        return
    else:
        st.success("✅ 知识库已加载，可以开始问答")

    # 在侧边栏或主界面添加图片上传功能
    def add_image_upload_section():
        """添加图片上传和识别功能"""
        with st.expander("📷 图片识别提问", expanded=False):
            st.info("上传包含问题的图片，系统会自动识别文字并提问")
            
            uploaded_file = st.file_uploader(
                "选择图片文件",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                help="支持PNG, JPG, JPEG, BMP格式"
            )
            
            if uploaded_file is not None:
                # 显示图片预览
                image_processor.display_image_preview(uploaded_file, "识别图片")
                
                # 检查是否已经识别过，避免重复识别
                if 'image_processed' not in st.session_state or st.session_state.get('last_uploaded_file') != uploaded_file.name:
                    # 识别文字
                    with st.spinner("正在识别图片中的文字..."):
                        result = image_processor.process_uploaded_image(uploaded_file)
                    
                    # 保存识别结果到session_state
                    st.session_state.image_processed = True
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.session_state.ocr_result = result
                else:
                    # 使用之前的结果
                    result = st.session_state.ocr_result
                
                # 处理识别结果
                if result["success"]:
                    st.success("✅ 文字识别成功！")
                    
                    # 显示识别结果并允许编辑
                    recognized_text = st.text_area(
                        "识别出的文字（可编辑）",
                        value=result["text"],
                        height=150,
                        help="检查并修改识别出的文字，然后点击'使用此文字提问'",
                        key="recognized_text"
                    )
                    
                    # 添加问题输入
                    additional_question = st.text_input(
                        "补充你的问题（可选）",
                        placeholder="例如：请解释这段话的意思...",
                        key="additional_question"
                    )
                    
                    # 提问按钮
                    if st.button("使用此文字提问", key="use_text_question"):
                        if recognized_text.strip():
                            # 组合问题和识别文字
                            full_question = recognized_text
                            if additional_question.strip():
                                full_question = f"{additional_question}\n\n识别内容：{recognized_text}"
                            
                            # 直接设置问题
                            st.session_state.image_question = full_question
                            
                            # 清除上传状态，避免重复识别
                            st.session_state.image_processed = False
                            st.session_state.last_uploaded_file = None
                            
                            st.success("✅ 问题已准备，系统正在自动处理...")
                        else:
                            st.error("请先识别出有效的文字")
                else:
                    st.error(f"❌ {result['message']}")
            
            # 添加强制重置按钮
            if st.button("重置图片上传", key="reset_upload"):
                st.session_state.image_processed = False
                st.session_state.last_uploaded_file = None
                st.session_state.image_question = ""
                st.rerun()


    add_image_upload_section()



    # ===== 正确的布局结构 =====
    if vector_store:
        # 创建两列布局
        col1, col2 = st.columns([2, 1])

        # 将聊天输入框移到最外层，确保始终显示
        user_input = st.chat_input("乖，你有哪个地方不明白呢")
        
        # 处理所有类型的问题输入
        current_prompt = None
        prompt_source = None

        # 首先检查是否有图片识别的问题
        if 'image_question' in st.session_state and st.session_state.image_question:
            current_prompt = st.session_state.image_question
            prompt_source = "image"
            # 立即清空图片问题，避免重复处理
            st.session_state.image_question = ""
        # 然后检查正常的聊天输入
        elif user_input:
            current_prompt = user_input
            prompt_source = "chat"


        # 如果有问题需要处理
        if current_prompt and st.session_state.deepseek_api.is_logged_in():
            # 用户消息
            st.session_state.messages.append({"role": "user", "content": current_prompt})
            
            # 助手回答
            with col1:
                with st.chat_message("assistant"):
                    with st.spinner("🔍 乖，稍等一下，让冰姐想一下这个问题..."):
                        try:
                            # 搜索文档
                            results = vector_store.search_similar_documents(current_prompt, k=10)
                            
                            if not results:
                                response = "乖，这个问题有点复杂，可以在课后答疑的时候问我，到时候冰姐语音给你讲哈"
                            else:
                                # 构建上下文 - 包含完整的元数据
                                contexts = []
                                for doc in results:
                                    contexts.append({
                                        'content': doc.page_content,
                                        'source': doc.metadata.get('source', '未知'),
                                        'page': doc.metadata.get('page', '未知页码')
                                    })

                                # 调用API，传递对话历史（排除当前这条用户消息）
                                conversation_history = st.session_state.messages[:-1]
                                response = st.session_state.deepseek_api.get_answer_with_new_template(
                                    current_prompt, contexts, conversation_history
                                )
                            
                            # 添加到消息历史
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                        except Exception as e:
                            error_msg = f"处理过程中出错: {str(e)}"
                            st.markdown(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

            # 如果是图片问题，显示成功提示
            if prompt_source == "image":
                st.success("✅ 图片问题已回答完成，可以继续提问")


        with col1:
            # 左侧：显示完整的聊天历史
            st.subheader("💬 对话界面")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])


        with col2:
            # 右侧：知识库状态（保持不变）
            st.subheader("📊 知识库状态")
            try:
                # 获取文档数量
                count = vector_store.vector_store._collection.count()
                
                # 显示统计卡片
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("总文本块数", f"{count:,}")
                with col_stat2:
                    st.metric("向量维度", "512")
                
                # 详细统计
                with st.expander("📈 详细统计", expanded=True):
                    st.info("**知识库信息:**")
                    st.write(f"• 文本块总数: **{count}** 个")
                    st.write(f"• 估算PDF文档: **~{max(1, floor(count // 80))}** 个")
                    st.write(f"• 向量维度: **512** 维")
                    st.write(f"• 检索模型: **BGE-small-ZH_v1.5**")
                    
                    # 进度条显示知识库规模
                    progress_value = min(count / 1000, 1.0)
                    st.progress(progress_value)
                    st.caption(f"知识库规模: {count}文本块")
                    
            except Exception as e:
                st.error(f"获取统计信息失败: {str(e)}")
    


    

if __name__ == "__main__":
    main()

