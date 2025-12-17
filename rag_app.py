import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 配置
load_dotenv()
DOCS_PATH = r"C:\Users\Administrator\Documents\EBOOK"  # 你的PDF目录
DB_PATH = "./chroma_db"  # 向量数据库保存位置

def get_vectorstore():
    # 使用本地 HuggingFace 向量模型（免费，无配额限制）
    print("--- 正在加载本地 Embedding 模型... ---")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 如果数据库已存在，直接加载
    if os.path.exists(DB_PATH):
        print("--- 正在从本地加载已有的向量数据库... ---")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        try:
            count = vectorstore._collection.count()
            print(f"--- 已加载数据库，当前包含 {count} 条文档分块 ---")
            if count > 0:
                return vectorstore
            print("--- ⚠️ 警告：数据库为空，准备重新扫描... ---")
        except Exception as e:
            print(f"--- ⚠️ 检查数据库状态时出错: {e}，准备重新扫描... ---")
    
    # 如果不存在或为空，则读取文件并创建
    print(f"--- 正在扫描目录 {DOCS_PATH} 中的 PDF 文件... ---")
    try:
        loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        raw_documents = loader.load()
        
        if not raw_documents:
            print(f"--- ⚠️ 警告：在 {DOCS_PATH} 中没有找到任何 PDF 文件！ ---")
            # 返回一个空的 vectorstore 或者抛出异常，这里我们创建一个空的
            return Chroma(embedding_function=embeddings, persist_directory=DB_PATH)

        print(f"--- 正在对 {len(raw_documents)} 页文档进行切分... ---")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(raw_documents)
        
        print(f"--- 正在生成向量并保存到 {DB_PATH}... (共 {len(documents)} 个分块) ---")
        
        # 本地模型可以一次性处理，无需分批
        vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=embeddings, 
            persist_directory=DB_PATH
        )
        
        print("--- 向量数据库创建完成！ ---")
        return vectorstore
    except Exception as e:
        print(f"--- ❌ 创建向量数据库时出错: {e} ---")
        raise e

def format_docs(docs):
    """格式化文档列表为字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    # 获取数据库
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        print(f"无法初始化数据库: {e}")
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 设置 DeepSeek 模型（用于生成回答）
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0
    )
    
    # 设置 RAG 提示模板
    system_prompt = """你是一个问答助手。使用以下检索到的上下文来回答问题。
如果你不知道答案，就说你不知道。保持回答简洁准确。

上下文：
{context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    # 使用 LCEL 构建 RAG 链
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n" + "="*50)
    print("🚀 RAG 系统已就绪！输入 'exit' 或 'quit' 退出。")
    print("="*50)

    # 交互循环
    while True:
        query = input("\n❓ 请输入你的问题: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("再见！")
            break
        
        if not query:
            continue

        print("🧠 思考中...")
        try:
            # 先获取相关文档用于显示来源
            docs = retriever.invoke(query)
            
            print(f"\n🔍 检索到 {len(docs)} 条相关记录:")
            for i, doc in enumerate(docs):
                source = os.path.basename(doc.metadata.get('source', '未知文件'))
                page = doc.metadata.get('page', '?')
                # 预览前100个字符，去除换行符以便显示
                content_preview = doc.page_content[:100].replace('\n', ' ') + "..."
                print(f"   [{i+1}] {source} (P{page}): {content_preview}")
            print("-" * 50)
            
            # 构建并打印完整的 prompt
            context = format_docs(docs)
            print("\n" + "="*50)
            print("📝 发送给大模型的完整 Prompt:")
            print("="*50)
            print("【系统提示】")
            print("你是一个问答助手。使用以下检索到的上下文来回答问题。")
            print("如果你不知道答案，就说你不知道。保持回答简洁准确。")
            print("\n上下文：")
            print(context)
            print("-"*50)
            print(f"【用户问题】{query}")
            print("="*50 + "\n")
            
            # 获取答案
            answer = rag_chain.invoke(query)

            print(f"\n🤖 AI 回答:\n{answer}")
            
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()