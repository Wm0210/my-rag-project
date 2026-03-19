import os
import re
import tempfile
import uuid
from datetime import datetime
from typing import List

import streamlit as st
import bs4
import oss2
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dashscope
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient
import hashlib
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct
from langchain_core.embeddings import Embeddings


class _DashScopeEmbeddingCore:
    """核心类：一次 API 调用（output_type='dense&sparse'）同时获取稠密和稀疏向量。

    内置短生命周期缓存（本次 Streamlit rerun 内有效），确保同一文本只调用一次 API，
    DashScopeEmbedder 和 DashScopeSparseEmbedder 共享此实例复用结果。
    """

    def __init__(self):
        self._cache: dict = {}  # key: (text, text_type) -> (dense, SparseVector)

    def fetch(self, text: str, text_type: str) -> tuple:
        """返回 (dense: List[float], sparse: SparseVector)，命中缓存则不重复调用。"""
        key = (text, text_type)
        if key not in self._cache:
            resp = dashscope.TextEmbedding.call(
                model="text-embedding-v4",
                input=text,
                output_type="dense&sparse",  # 一次调用同时返回稠密+稀疏
                text_type=text_type,         # "query"/"document" 分别针对检索侧优化
                api_key=st.session_state.dashscope_api_key,
            )
            if resp.status_code != 200:
                raise ValueError(f"DashScope embedding failed: {resp.message}")
            emb = resp.output["embeddings"][0]
            dense = emb["embedding"]                       # List[float], 1024 维
            sv_raw = emb["sparse_embedding"]               # List[{"index", "token", "value"}]
            sparse = SparseVector(
                indices=[int(item["index"]) for item in sv_raw],
                values=[float(item["value"]) for item in sv_raw],
            )
            self._cache[key] = (dense, sparse)
        return self._cache[key]


class DashScopeEmbedder(Embeddings):
    """稠密向量 adapter，实现 LangChain Embeddings 接口，与 DashScopeSparseEmbedder 共享核心。"""

    def __init__(self, core: _DashScopeEmbeddingCore):
        self._core = core

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._core.fetch(t, "document")[0] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._core.fetch(text, "query")[0]


class DashScopeSparseEmbedder(SparseEmbeddings):
    """稀疏向量 adapter，实现 langchain_qdrant SparseEmbeddings 接口，与 DashScopeEmbedder 共享核心。"""

    def __init__(self, core: _DashScopeEmbeddingCore):
        self._core = core

    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        return [self._core.fetch(t, "document")[1] for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._core.fetch(text, "query")[1]


# Constants
COLLECTION_NAME = "qwen-rag-agent"
METADATA_COLLECTION_NAME = "qwen-rag-metadata"  # 持久化已解析文件列表的元数据 collection

# Streamlit App Initialization
st.title("🤔 RAG助手") # 标题

# Session State Initialization
if "dashscope_api_key" not in st.session_state:
    st.session_state.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
if "qdrant_url" not in st.session_state:
    st.session_state.qdrant_url = "http://localhost:6333"
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []
if "history" not in st.session_state:
    st.session_state.history = []
if "exa_api_key" not in st.session_state:
    st.session_state.exa_api_key = os.getenv("EXA_API_KEY", "")
if "use_web_search" not in st.session_state:
    st.session_state.use_web_search = True
if "force_web_search" not in st.session_state:
    st.session_state.force_web_search = False
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.8
if "use_query_rewrite" not in st.session_state:
    st.session_state.use_query_rewrite = False
if "use_memory" not in st.session_state:
    st.session_state.use_memory = True
if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "向量检索"
if "processed_documents_loaded" not in st.session_state:
    st.session_state.processed_documents_loaded = False
if "cited_docs" not in st.session_state:
    st.session_state.cited_docs = {}  # 当前轮次的编号→doc映射，供来源面板使用
if "oss_access_key_id" not in st.session_state:
    st.session_state.oss_access_key_id = os.getenv("ACCESS_KEY_ID", "")
if "oss_access_key_secret" not in st.session_state:
    st.session_state.oss_access_key_secret = os.getenv("ACCESS_KEY_SECRET", "")
if "oss_endpoint" not in st.session_state:
    st.session_state.oss_endpoint = os.getenv("ENDPOINT", "")
if "oss_bucket_name" not in st.session_state:
    st.session_state.oss_bucket_name = os.getenv("BUCKET_NAME", "")


# Sidebar Configuration
st.sidebar.header("🔑 API 设置")
dashscope_api_key = st.sidebar.text_input(
    "DashScope API Key", type="password", value=st.session_state.dashscope_api_key
)
qdrant_url = st.sidebar.text_input(
    "Qdrant URL (Local Docker)",
    placeholder="http://localhost:6333", # 框内隐藏提示
    value=st.session_state.qdrant_url,
)

# OSS Configuration
with st.sidebar.expander("🗄️ 阿里云 OSS 设置（图片上传必填）"):
    st.session_state.oss_access_key_id = st.text_input(
        "Access Key ID", type="password", value=st.session_state.oss_access_key_id, key="_oss_akid"
    )
    st.session_state.oss_access_key_secret = st.text_input(
        "Access Key Secret", type="password", value=st.session_state.oss_access_key_secret, key="_oss_aksec"
    )
    st.session_state.oss_endpoint = st.text_input(
        "Endpoint", placeholder="oss-cn-hangzhou.aliyuncs.com",
        value=st.session_state.oss_endpoint, key="_oss_ep"
    )
    st.session_state.oss_bucket_name = st.text_input(
        "Bucket Name", value=st.session_state.oss_bucket_name, key="_oss_bkt"
    )

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.rerun()

# Update session state
st.session_state.dashscope_api_key = dashscope_api_key
st.session_state.qdrant_url = qdrant_url

# Web Search Configuration
# st.sidebar.header("🌐 Web Search Configuration")
st.sidebar.header("🌐 联网搜索")
st.session_state.use_web_search = st.sidebar.checkbox(
    "Enable Web Search Fallback", value=st.session_state.use_web_search
)

search_domains: List[str] = []  # default; overridden when web search is enabled
if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input(
        "Exa AI API Key",
        type="password",
        value=st.session_state.exa_api_key,
        help="Required for web search fallback when no relevant documents are found",
    )
    st.session_state.exa_api_key = exa_api_key

    default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    custom_domains = st.sidebar.text_input(
        "Custom domains (comma-separated)",
        value=",".join(default_domains),
        help="Enter domains to search from, e.g.: arxiv.org,wikipedia.org",
    )
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

# Search Configuration
st.sidebar.header("🎯 搜索设置")
_mode_options = ["向量检索", "全文检索", "混合检索"]
st.session_state.retrieval_mode = st.sidebar.radio(
    "检索模式",
    _mode_options,
    index=_mode_options.index(st.session_state.retrieval_mode),
    help="向量：语义相似度 | 全文：关键词精确匹配 | 混合：向量+全文 RRF融合（推荐）",
)
st.session_state.similarity_threshold = st.sidebar.slider(
    # "Document Similarity Threshold",
    "文档相似度阈值",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    # help="Lower values return more docs but may be less relevant.",
    help="较低的值会返回更多文档，但可能相关性较差。",
)
st.session_state.use_query_rewrite = st.sidebar.checkbox(
    "启用查询重写", value=st.session_state.use_query_rewrite,
    help="开启后，每次提问会先用 LLM 改写问题以提升检索质量，但会多消耗一次 LLM 调用。"
)
st.session_state.use_memory = st.sidebar.checkbox(
    "启用记忆（最近3轮）", value=st.session_state.use_memory,
    help="开启后，最近 3 轮对话记录将作为上下文参与查询重写和最终回答生成。"
)

# ─── Utility Functions ────────────────────────────────────────────────────────

@st.cache_resource
def init_qdrant(qdrant_url: str):
    """Initialize Qdrant client for local Docker deployment."""
    if not qdrant_url:
        return None
    try:
        client = QdrantClient(url=qdrant_url, timeout=60)
        client.get_collections()  # connection test
        return client
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None


# ─── Document Processing ──────────────────────────────────────────────────────

def process_pdf(file) -> List:
    """Process PDF file and add source metadata."""
    tmp_path = "" # 初始化变量，用于记录临时文件路径
    try:
        # 1. 桥接内存与硬盘
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name  # 记录下这个克隆体的绝对路径

        # 2. 加载与解析 (使用更快的 PyMuPDF)
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()

        # 3. 注入元数据
        for doc in documents:
            doc.metadata.update(
                {
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # 4. 文本切片
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        
        # 5. [最佳实践] “过河拆桥”：解析完成后，立刻删掉临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        return split_docs

    except Exception as e:
        # 如果出错，也尽量把临时文件删掉
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        st.error(f"📄 PDF processing error: {str(e)}")
        return []


def process_web(url: str) -> List:
    """Process web URL using Unstructured to intelligently extract main content."""
    try:
        # 1. 使用 UnstructuredURLLoader 替代 WebBaseLoader
        # 它的底层算法会自动识别网页主体结构，剔除页眉、页脚和侧边栏
        loader = UnstructuredURLLoader(urls=[url])
        # loader = WebBaseLoader(
        #     web_paths=(url,),
        #     # bs_kwargs=dict(
        #     #     parse_only=bs4.SoupStrainer(
        #     #         class_=("post-content", "post-title", "post-header", "content", "main")
        #     #     )
        #     # ),
        #     # SoupStrainer 仅匹配特定博客 class，对大多数网站会返回空内容，已禁用
        # )
        documents = loader.load()

        for doc in documents:
            # 2. 轻量级文本清洗
            # Unstructured 提取的文本已经很干净了，但有时会保留多余的连续换行
            # 这里将 3 个以上的连续换行符统一替换为 2 个换行符，保持段落清晰
            clean_text = re.sub(r'\n{3,}', '\n\n', doc.page_content)
            doc.page_content = clean_text.strip()
            
            # 3. 注入元数据
            doc.metadata.update(
                {
                    "source_type": "url",
                    "url": url,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # 4. 文本切分
        # 增加分隔符层级（优先按段落切，再按句子切），防止切断上下文逻辑
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100,
            # separators=["\n\n", "\n", "。", "！", "？", " ", ""] 
        )
        return text_splitter.split_documents(documents)
    
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []

# ─── Image Processing ────────────────────────────────────────────

def upload_to_oss(file_bytes: bytes, file_name: str) -> str:
    """上传图片到阿里云 OSS，返回公网可访问 URL。"""
    auth = oss2.Auth(
        st.session_state.oss_access_key_id,
        st.session_state.oss_access_key_secret,
    )
    bucket = oss2.Bucket(
        auth, st.session_state.oss_endpoint, st.session_state.oss_bucket_name
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_name = f"{timestamp}_{uuid.uuid4().hex}_{file_name}"
    object_key = f"qwen-rag/{unique_name}"
    ext = file_name.lower().rsplit(".", 1)[-1]
    content_type_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "webp": "image/webp",
    }
    result = bucket.put_object(
        object_key, file_bytes,
        headers={"Content-Type": content_type_map.get(ext, "image/jpeg")},
    )
    if result.status == 200:
        return f"https://{st.session_state.oss_bucket_name}.{st.session_state.oss_endpoint}/{object_key}"
    raise Exception(f"OSS 上传失败，状态码: {result.status}")


def describe_image_with_vl(image_url: str) -> str:
    """调用 qwen3-vl-flash（非思考模式）生成图片的详细检索描述。"""
    prompt = (
        "请对这张图片进行详细描述，内容包括：\n"
        "1. 图片的整体场景和主题\n"
        "2. 图中出现的主要人物、物体、标志或元素\n"
        "3. 图中如有文字，请完整转录\n"
        "4. 图片的颜色、风格、构图特点\n"
        "5. 图片传达的主要信息或意图\n"
        "请用中文详细描述，尽量提取有助于后续检索的关键词和细节信息。"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": prompt},
            ],
        }
    ]
    response = dashscope.MultiModalConversation.call(
        api_key=st.session_state.dashscope_api_key,
        model="qwen3-vl-flash",
        messages=messages,
        stream=False,
        enable_thinking=False,
    )
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    raise Exception(f"qwen3-vl-flash 调用失败: {response.message}")


def process_image(file) -> List:
    """图片 RAG 入库流程：OSS 上传 → VL 描述 → 封装 Document。"""
    from langchain_core.documents import Document
    try:
        image_bytes = file.getvalue()
        file_name = file.name

        # Step 1: 上传到 OSS
        with st.status("📤 正在上传图片到阿里云 OSS…", expanded=True) as status:
            oss_url = upload_to_oss(image_bytes, file_name)
            st.write(f"✅ 上传成功: `{oss_url}`")
            status.update(label="✅ OSS 上传完成", state="complete", expanded=False)

        # Step 2: 调用 qwen3-vl-flash 生成图片描述
        with st.status("🔍 qwen3-vl-flash 正在理解图片内容…", expanded=True) as status:
            description = describe_image_with_vl(oss_url)
            preview = description[:120].replace("\n", " ")
            st.write(f"📝 描述预览：{preview}…")
            status.update(label="✅ 视觉理解完成", state="complete", expanded=False)

        # Step 3: 封装为 LangChain Document
        doc = Document(
            page_content=description,
            metadata={
                "source_type": "image",
                "file_name": file_name,
                "oss_url": oss_url,
                "timestamp": datetime.now().isoformat(),
            },
        )
        return [doc]

    except Exception as e:
        st.error(f"🖼️ 图片处理失败: {str(e)}")
        return []



# ─── Excel Processing ─────────────────────────────────────────────────────────

def _df_to_markdown(df) -> str:
    """将 DataFrame 转换为 Markdown 表格字符串（无需 tabulate 依赖）。"""
    cols = [str(c) for c in df.columns.tolist()]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(
            "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
            for v in row
        ) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep] + rows)


def analyze_table_with_llm(md_table: str, sheet_name: str, file_name: str) -> str:
    """调用 qwen3.5-plus 对表格进行语义分析，生成便于向量检索的描述文本。"""
    llm = get_llm()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一位数据分析专家，擅长理解表格数据并生成准确的语义摘要。"),
        ("human",
         "请对以下来自文件「{file_name}」中 Sheet「{sheet_name}」的表格进行语义分析，内容包括：\n"
         "1. 表格的整体主题和业务含义\n"
         "2. 各列的含义和数据特征（数据类型、取值范围等）\n"
         "3. 数据中的关键规律、趋势或特征值\n"
         "4. 表格的潜在用途或应用场景\n"
         "5. 提取便于检索的关键词和核心概念\n\n"
         "表格内容：\n{md_table}\n\n"
         "请用中文详细描述，重点提取有助于语义检索的信息。"),
    ])
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"file_name": file_name, "sheet_name": sheet_name, "md_table": md_table})


def process_excel(file) -> List:
    """Excel RAG 入库流程：解析各 Sheet → Markdown → LLM 语义分析 → 封装 Document。"""
    from langchain_core.documents import Document
    import io
    try:
        file_name = file.name
        file_bytes = file.getvalue()
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet_names = xl.sheet_names
        docs = []
        for sheet_name in sheet_names:
            df = xl.parse(sheet_name)
            if df.empty:
                continue
            md_table = _df_to_markdown(df)
            columns = [str(c) for c in df.columns.tolist()]
            row_count = len(df)
            # LLM 语义分析（page_content 用于向量化检索，md_table 留给回答阶段）
            with st.status(f"🤖 正在分析 Sheet「{sheet_name}」（{row_count} 行 × {len(columns)} 列）…", expanded=True) as status:
                analysis = analyze_table_with_llm(md_table, sheet_name, file_name)
                preview = analysis[:100].replace("\n", " ")
                st.write(f"📝 分析预览：{preview}…")
                status.update(label=f"✅ Sheet「{sheet_name}」分析完成", state="complete", expanded=False)
            doc = Document(
                page_content=analysis,          # 语义分析文本 → 向量化
                metadata={
                    "source_type": "excel",
                    "file_name": file_name,
                    "sheet_name": sheet_name,
                    "md_table": md_table,       # 原始 Markdown 表格 → 回答时使用
                    "columns": columns,
                    "row_count": row_count,
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                },
            )
            docs.append(doc)
        return docs
    except Exception as e:
        st.error(f"📊 Excel 处理失败: {str(e)}")
        return []


# ─── Vector Store ─────────────────────────────────────────────────────────────

def get_vector_store(client, retrieval_mode: str = "混合检索") -> QdrantVectorStore:
    """根据检索模式创建 QdrantVectorStore，稠密+稀疏双路嵌入器始终绑定。

    上传文档时固定使用 '混合检索'（HYBRID）模式，确保两路向量都写入。
    查询时按用户选择的模式动态切换 RetrievalMode。
    """
    mode_map = {
        "向量检索": RetrievalMode.DENSE,
        "全文检索": RetrievalMode.SPARSE,
        "混合检索": RetrievalMode.HYBRID,
    }
    core = _DashScopeEmbeddingCore()  # 共享核心：两个 adapter 各拿 dense/sparse，只调一次 API
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=DashScopeEmbedder(core),
        sparse_embedding=DashScopeSparseEmbedder(core),
        vector_name="dense",
        sparse_vector_name="sparse",
        retrieval_mode=mode_map.get(retrieval_mode, RetrievalMode.HYBRID),
    )


def create_vector_store(client, texts):
    """创建支持稠密+稀疏双向量的 collection 并写入文档。"""
    try:
        try:
            # 创建双向量 collection：dense（稠密）+ sparse（稀疏）
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(
                        size=1024,  # text-embedding-v4 维度
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)  # 稀疏索引保留在内存，查询更快
                    )
                },
            )
            st.success(f"📚 Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e

        # 使用混合模式写入，确保两路向量都被计算并存入
        vector_store = get_vector_store(client, "混合检索")

        with st.spinner("Uploading documents to Qdrant..."):
            vector_store.add_documents(texts)
            st.success("✅ Documents stored successfully!")
            return vector_store

    except Exception as e:
        st.error(f"🔴 Vector store error: {str(e)}")
        return None




# ─── Metadata Store（持久化已解析文件列表）────────────────────────────────────

def ensure_metadata_collection(client: QdrantClient):
    """确保元数据 collection 存在（1 维哑向量，仅靠 payload 存储文件名）。"""
    existing = [c.name for c in client.get_collections().collections]
    if METADATA_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=METADATA_COLLECTION_NAME,
            vectors_config=VectorParams(size=1, distance=Distance.DOT),
        )


def record_processed_source(client: QdrantClient, source_name: str):
    """将已解析的文件名/URL 写入元数据 collection，用名称 MD5 作 point ID（幂等）。"""
    ensure_metadata_collection(client)
    point_id = int(hashlib.md5(source_name.encode()).hexdigest(), 16) % (2 ** 63)
    client.upsert(
        collection_name=METADATA_COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=[1.0],
                payload={
                    "source": source_name,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ],
    )


def load_processed_sources(client: QdrantClient) -> list:
    """从元数据 collection 读取所有已解析文件名/URL（分页滚动，支持大量文件）。"""
    ensure_metadata_collection(client)
    sources = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=METADATA_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            if "source" in p.payload:
                sources.append(p.payload["source"])
        if offset is None:
            break
    return sources

# ─── LLM & Agent Functions (LangChain) ───────────────────────────────────────



def format_memory(history: list, n_rounds: int = 3) -> str:
    """从历史记录中提取最近 n_rounds 轮对话（用户问题 + 模型回答）格式化为字符串。
    
    Returns 空字符串（无历史）或格式化后的多轮对话文本。
    """
    pairs = []
    temp_user = None
    for msg in history:
        if msg["role"] == "user":
            temp_user = msg["content"]
        elif msg["role"] == "assistant" and temp_user is not None:
            pairs.append((temp_user, msg["content"]))
            temp_user = None
    recent = pairs[-n_rounds:]
    if not recent:
        return ""
    lines = []
    for idx, (q, a) in enumerate(recent, 1):
        lines.append(f"第{idx}轮\n用户：{q}\n助手：{a}")
    return "\n\n".join(lines)


def get_llm() -> ChatOpenAI:
    """返回配置好的 DashScope Qwen LLM 实例。"""
    return ChatOpenAI(
        model="qwen3.5-plus",
        api_key=st.session_state.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": False},
    )


def rewrite_query(user_query: str, memory_context: str = "") -> str:
    """使用 LangChain 链将用户问题改写为更精确的检索语句。"""
    llm = get_llm()
    human_content = (
        f"历史对话记录（最近几轮，供参考）：\n{memory_context}\n\n当前问题：{user_query}"
        if memory_context else user_query
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at reformulating questions to be more precise and detailed.
        Your task is to:
        1. Analyze the user's question
        2. Rewrite it to be more specific and search-friendly
        3. Expand any acronyms or technical terms
        4. If conversation history is provided, use it to resolve pronouns, references, or follow-up context
        5. Return ONLY the rewritten query without any additional text or explanations

        Example 1:
        User: "What does it say about ML?"
        Output: "What are the key concepts, techniques, and applications of Machine Learning (ML) discussed in the context?"

        Example 2:
        User: "Tell me about transformers"
        Output: "Explain the architecture, mechanisms, and applications of Transformer neural networks in natural language processing and deep learning"
        
        If the user asks in Chinese, please return the rewritten query in Chinese.
        """,
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"query": human_content})


def run_web_search(query: str, domains: List[str]) -> tuple:
    """使用 Exa 搜索网络，并用 LLM 整理摘要返回，同时返回原始来源列表。

    Returns:
        (summary_str, sources_list)
        sources_list: [{"url": ..., "title": ..., "content": ...}, ...]
    """
    try:
        from exa_py import Exa  # pip install exa-py

        exa = Exa(api_key=st.session_state.exa_api_key)
        search_results = exa.search(
            query,
            num_results=5,
            # include_domains=domains if domains else None,
            contents={"text": {"max_characters": 3000}},
            # user_location="CN",
        )

        results_text = ""
        sources = []
        for r in search_results.results:
            results_text += (
                f"Source: {r.url}\nTitle: {r.title}\nContent: {r.text}\n\n"
            )
            sources.append({
                "url": r.url,
                "title": r.title or r.url,
                "content": r.text or "",
            })

        if not results_text:
            return "No relevant web search results found.", []

        llm = get_llm()
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a web search expert. Compile and summarize the following search results clearly. Include source URLs in your response.",
                ),
                ("human", "Query: {query}\n\nSearch Results:\n{results}"),
            ]
        )
        chain = prompt_template | llm | StrOutputParser()
        summary = chain.invoke({"query": query, "results": results_text})
        return summary, sources
    except Exception as e:
        return f"Web search error: {str(e)}", []


def generate_rag_response(full_prompt: str, memory_context: str = "") -> str:
    """使用 Qwen LLM 生成 RAG 最终回答。"""
    llm = get_llm()
    actual_prompt = (
        f"对话历史记忆（最近几轮问答，供参考）：\n{memory_context}\n\n{full_prompt}"
        if memory_context else full_prompt
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an Intelligent Agent specializing in providing accurate answers.

        When given context from documents:
        - Focus on information from the provided documents
        - Be precise and cite specific details
        - CRITICAL: When you use information from a numbered document chunk [N], cite it inline using the format [N] at the end of the relevant sentence. For example: "迪士尼乐园开放时间为9:00-22:00 [1]。"
        - Cite all source chunks that contribute to each statement. If multiple chunks support one point, list them together: [1][2].

        When given web search results:
        - Clearly indicate that the information comes from web search
        - Synthesize the information clearly

        When given conversation history:
        - Use it to understand the context and provide coherent follow-up answers

        - Please respond in the same language as the user's question.
        
        Always maintain high accuracy and clarity in your responses.""",
            ),
            ("human", "{prompt}"),
        ]
    )
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"prompt": actual_prompt})




def generate_multimodal_response(query: str, image_docs: List, text_context: str, memory_context: str = "") -> str:
    """检索到图像块时，调用 qwen3.5-plus 多模态模型，将 OSS 原图作为视觉上下文回答问题。"""
    content = []
    # 将所有相关图片的 OSS URL 注入消息
    for doc in image_docs:
        oss_url = doc.metadata.get("oss_url", "")
        if oss_url:
            content.append({"image": oss_url})
    # 构建文字部分
    text_parts = []
    if memory_context:
        text_parts.append(f"对话历史记忆（最近几轮问答，供参考）：\n{memory_context}")
    if text_context:
        text_parts.append(f"以下是检索到的相关文档内容：\n{text_context}")
    text_parts.append(
        f"用户问题：{query}\n\n"
        "请根据以上图片和文档内容，用与用户问题相同的语言给出详细、准确的回答。"
        "若答案主要来自图片，请指出具体图片内容；若来自文档，请注明相关依据。"
    )
    content.append({"text": "\n\n".join(text_parts)})
    messages = [{"role": "user", "content": content}]
    response = dashscope.MultiModalConversation.call(
        api_key=st.session_state.dashscope_api_key,
        model="qwen3.5-plus",
        messages=messages,
        stream=False,
    )
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    raise Exception(f"qwen3.5-plus 多模态调用失败: {response.message}")


def rerank_documents(query: str, docs: List, top_n: int = 3) -> List:
    """使用 qwen3-rerank 对粗排结果进行精排，返回最相关的 top_n 个文档。"""
    if not docs:
        return docs
    try:
        resp = dashscope.TextReRank.call(
            model="qwen3-rerank",
            query=query,
            documents=[doc.page_content for doc in docs],
            top_n=top_n,
            api_key=st.session_state.dashscope_api_key,
            instruct="Given a web search query, retrieve relevant passages that answer the query.",
        )
        if resp.status_code != 200:
            st.warning(f"⚠️ 重排序失败：{resp.message}，使用原始排序。")
            return docs[:top_n]
        # results 已按 relevance_score 从高到低排好序，通过 index 映射回原始 LangChain Document
        reranked = [docs[r["index"]] for r in resp.output["results"]]
        return reranked
    except Exception as e:
        st.warning(f"⚠️ 重排序出错：{str(e)}，使用原始排序。")
        return docs[:top_n]


def _get_source_label(cite_num: int, doc) -> str:
    """根据 doc 类型返回来源摘要标签（用于 expander 标题）。"""
    source_type = doc.metadata.get("source_type", "unknown")
    if source_type == "pdf":
        fname = doc.metadata.get("file_name", "未知文件")
        page = doc.metadata.get("page", None)
        page_str = f"  第 {page + 1} 页" if page is not None else ""
        return f"{fname}{page_str}"
    elif source_type == "excel":
        fname = doc.metadata.get("file_name", "未知文件")
        sheet = doc.metadata.get("sheet_name", "")
        rows = doc.metadata.get("row_count", "")
        return f"{fname} / Sheet「{sheet}」{f'（{rows} 行）' if rows else ''}"
    elif source_type == "image":
        return doc.metadata.get("file_name", "未知图片")
    else:
        url = doc.metadata.get("url", "")
        return url[:60] + ("…" if len(url) > 60 else "")


def _render_source_card(cite_num: int, doc):
    """在 expander 内渲染来源详情卡片。"""
    source_type = doc.metadata.get("source_type", "unknown")
    if source_type == "pdf":
        fname = doc.metadata.get("file_name", "未知文件")
        page = doc.metadata.get("page", None)
        page_str = f"第 {page + 1} 页" if page is not None else "页码未知"
        st.markdown(f"📄 **{fname}** — {page_str}")
        st.markdown("**块内容：**")
        st.info(doc.page_content)
    elif source_type == "excel":
        fname = doc.metadata.get("file_name", "未知文件")
        sheet = doc.metadata.get("sheet_name", "")
        cols = doc.metadata.get("columns", [])
        rows = doc.metadata.get("row_count", 0)
        md_table = doc.metadata.get("md_table", "")
        st.markdown(f"📊 **{fname}** — Sheet「{sheet}」（{rows} 行）")
        if cols:
            st.caption(f"列名：{', '.join(cols)}")
        if md_table:
            md_lines = md_table.split("\n")
            st.markdown("**表格预览（前12行）：**")
            st.markdown("\n".join(md_lines[:12]))
        st.markdown("**语义分析：**")
        st.caption(doc.page_content[:300] + ("…" if len(doc.page_content) > 300 else ""))
    elif source_type == "image":
        fname = doc.metadata.get("file_name", "未知图片")
        oss_url = doc.metadata.get("oss_url", "")
        st.markdown(f"🖼️ **{fname}**")
        if oss_url:
            st.image(oss_url, caption=fname, width=320)
        st.markdown("**VL 描述：**")
        st.info(doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""))
    else:
        url = doc.metadata.get("url", "")
        st.markdown(f"🌐 **来源 URL：** [{url}]({url})")
        st.markdown("**块内容：**")
        st.info(doc.page_content)


# ─── Main Application Flow ────────────────────────────────────────────────────

if st.session_state.dashscope_api_key:
    os.environ["DASHSCOPE_API_KEY"] = st.session_state.dashscope_api_key

    # 初始化 Qdrant 客户端
    qdrant_client = init_qdrant(st.session_state.qdrant_url) 

    # 从元数据 collection 恢复已解析文件列表（每次 session 只加载一次）
    if qdrant_client and not st.session_state.processed_documents_loaded:
        st.session_state.processed_documents = load_processed_sources(qdrant_client)
        st.session_state.processed_documents_loaded = True

    # Auto-reconnect to existing Qdrant collection on restart 
    # 连接到现有的 Qdrant 集合（如果存在）以保持数据持久性
    if qdrant_client and st.session_state.vector_store is None:
        existing = [c.name for c in qdrant_client.get_collections().collections]
        if COLLECTION_NAME in existing:
            # 以混合模式连接，确保上传新文档时两路向量都能写入
            st.session_state.vector_store = get_vector_store(qdrant_client, "混合检索")
            st.sidebar.success(f"✅ Reconnected to existing collection: {COLLECTION_NAME}")

    # File/URL Upload Section
    st.sidebar.header("📁 上传文件（PDF / 图片）")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    uploaded_image = st.sidebar.file_uploader(
        "Upload Image（JPG / PNG / WEBP）",
        type=["jpg", "jpeg", "png", "webp"],
        help="图片将上传至阿里云 OSS，由 qwen3-vl-flash 生成文本描述后嵌入向量库",
    )
    uploaded_excel = st.sidebar.file_uploader(
        "Upload Excel（XLS / XLSX）",
        type=["xls", "xlsx"],
        help="各 Sheet 独立处理：转 Markdown → qwen3.5-plus 语义分析 → 向量化入库",
    )
    web_url = st.sidebar.text_input("Or enter URL", placeholder="https://...")
    add_url = st.sidebar.button("➕ 添加 URL")

    # Process PDF
    if uploaded_file:
        file_name = uploaded_file.name # 目前只起到本次运行内去重的作用，后续可以改为更健壮的哈希值或数据库记录
        if file_name not in st.session_state.processed_documents:
            with st.spinner("Processing PDF..."):
                texts = process_pdf(uploaded_file) # 处理 PDF 文件并返回切分后的文本块列表，每个文本块都带有元数据（如来源、时间戳等）
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts) # 向量化并添加到现有集合（向量库）
                    else:
                        st.session_state.vector_store = create_vector_store(
                            qdrant_client, texts
                        )
                    st.session_state.processed_documents.append(file_name)
                    record_processed_source(qdrant_client, file_name)
                    st.success(f"✅ Added PDF: {file_name}")

    # Process Image
    if uploaded_image:
        img_name = uploaded_image.name
        if img_name not in st.session_state.processed_documents:
            oss_cfg_ok = all([
                st.session_state.oss_access_key_id,
                st.session_state.oss_access_key_secret,
                st.session_state.oss_endpoint,
                st.session_state.oss_bucket_name,
            ])
            if not oss_cfg_ok:
                st.sidebar.warning("⚠️ 请先在侧边栏展开「阿里云 OSS 设置」并填写完整配置。")
            else:
                docs = process_image(uploaded_image)
                if docs and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(docs)
                    else:
                        st.session_state.vector_store = create_vector_store(qdrant_client, docs)
                    st.session_state.processed_documents.append(img_name)
                    record_processed_source(qdrant_client, img_name)
                    st.success(f"✅ 图片已入库: {img_name}")
        else:
            st.sidebar.info(f"ℹ️ {img_name} 已入库，无需重复处理。")

    # Process Excel
    if uploaded_excel:
        excel_name = uploaded_excel.name
        if excel_name not in st.session_state.processed_documents:
            _excel_docs = process_excel(uploaded_excel)
            if _excel_docs and qdrant_client:
                if st.session_state.vector_store:
                    st.session_state.vector_store.add_documents(_excel_docs)
                else:
                    st.session_state.vector_store = create_vector_store(qdrant_client, _excel_docs)
                st.session_state.processed_documents.append(excel_name)
                record_processed_source(qdrant_client, excel_name)
                st.success(f"✅ 表格已入库: {excel_name}（共 {len(_excel_docs)} 个 Sheet）")
        else:
            st.sidebar.info(f"ℹ️ {excel_name} 已入库，无需重复处理。")

    if add_url and web_url:
        if web_url not in st.session_state.processed_documents:
            with st.spinner("Processing URL..."):
                texts = process_web(web_url)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(
                            qdrant_client, texts
                        )
                    st.session_state.processed_documents.append(web_url)
                    record_processed_source(qdrant_client, web_url)
                    st.success(f"✅ Added URL: {web_url}")
                elif not texts:
                    st.sidebar.warning("⚠️ 未能从该 URL 提取到任何文本，请检查链接或网页内容。")
        else:
            st.sidebar.info(f"ℹ️ 该 URL 已入库，无需重复添加。")

    # Display sources in sidebar
    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        for source in st.session_state.processed_documents:
            if source.endswith(".pdf"):
                st.sidebar.text(f"📄 {source}")
            elif source.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                st.sidebar.text(f"🖼️ {source}")
            elif source.lower().endswith((".xlsx", ".xls")):
                st.sidebar.text(f"📊 {source}")
            else:
                st.sidebar.text(f"🌐 {source}")

    # Chat Interface
    # Render chat history on every rerun
    for message in st.session_state.history:
        with st.chat_message(message["role"]): # 角色气泡
            st.write(message["content"]) # 消息内容

    chat_col, toggle_col = st.columns([0.9, 0.1]) # 纵向布局：输入框占90%，开关占10%
    with chat_col:
        prompt = st.chat_input("Ask about your documents...")
    with toggle_col:
        st.session_state.force_web_search = st.toggle("🌐", help="Force web search")

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 每轮对话开始时重置引用映射
        st.session_state.cited_docs = {}

        # 计算记忆上下文（取当前消息之前的历史轮次，最多 3 轮）
        memory_context = ""
        if st.session_state.use_memory:
            memory_context = format_memory(st.session_state.history[:-1])

        # Step 1: Rewrite the query for better retrieval (optional)
        if st.session_state.use_query_rewrite:
            with st.spinner("🤔 Reformulating query..."):
                try:
                    rewritten_query = rewrite_query(prompt, memory_context=memory_context)
                    with st.expander("🔄 See rewritten query"): # 折叠框内展示原始prompt以及改写之后的
                        st.write(f"Original: {prompt}")
                        st.write(f"Rewritten: {rewritten_query}")
                except Exception as e:
                    st.error(f"❌ Error rewriting query: {str(e)}")
                    rewritten_query = prompt
        else:
            rewritten_query = prompt

        # Step 2: Document retrieval
        context = ""
        docs = []
        image_docs = []
        excel_docs = []
        web_sources = []
        if not st.session_state.force_web_search and st.session_state.vector_store:
            # 根据用户选择的检索模式动态创建检索器
            store = get_vector_store(qdrant_client, st.session_state.retrieval_mode)
            if st.session_state.retrieval_mode == "向量检索":
                # 向量检索：支持相似度阈值过滤
                retriever = store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": 15,
                        "score_threshold": st.session_state.similarity_threshold,
                    },
                )
            else:
                # 全文/混合检索：BM25/RRF 分数量纲不同，直接取 top-k
                retriever = store.as_retriever(search_kwargs={"k": 15})
            docs = retriever.invoke(rewritten_query)  # 每个元素包含 page_content 和 metadata
            if docs:
                retrieved_count = len(docs)
                with st.spinner("🔄 重排序中（qwen3-rerank）..."):
                    docs = rerank_documents(rewritten_query, docs, top_n=5)
                # 按来源类型分流
                image_docs = [d for d in docs if d.metadata.get("source_type") == "image"]
                excel_docs = [d for d in docs if d.metadata.get("source_type") == "excel"]
                text_docs  = [d for d in docs if d.metadata.get("source_type") not in ("image", "excel")]
                # 构建带编号的 context，同时记录编号→doc映射
                cited_docs = {}
                context_parts = []
                _cite_idx = 1
                for d in text_docs:
                    cited_docs[_cite_idx] = d
                    context_parts.append(f"[{_cite_idx}] {d.page_content}")
                    _cite_idx += 1
                # 表格块：检索时用语义分析召回，回答时改用原始 Markdown 表格
                for d in excel_docs:
                    cited_docs[_cite_idx] = d
                    context_parts.append(
                        f"[{_cite_idx}] [表格「{d.metadata.get('file_name', '未知')}」- Sheet:「{d.metadata.get('sheet_name', '')}」]\n{d.metadata.get('md_table', d.page_content)}"
                    )
                    _cite_idx += 1
                # 图片块：回答时走多模态路径，此处 VL 描述仅作文字备注
                for d in image_docs:
                    cited_docs[_cite_idx] = d
                    context_parts.append(
                        f"[{_cite_idx}] [图片「{d.metadata.get('file_name', '未知')}」视觉描述]\n{d.page_content}"
                    )
                    _cite_idx += 1
                context = "\n\n".join(context_parts)
                st.session_state.cited_docs = cited_docs
                mode_icon = {"向量检索": "🔮", "全文检索": "��", "混合检索": "⚡"}
                icon = mode_icon.get(st.session_state.retrieval_mode, "📊")
                hints = []
                if image_docs:
                    hints.append(f"{len(image_docs)} 张图片")
                if excel_docs:
                    hints.append(f"{len(excel_docs)} 个表格")
                img_hint = f"（含 {'、'.join(hints)}）" if hints else ""
                st.info(f"{icon} 粗排召回 {retrieved_count} 个文本块，重排序后保留 {len(docs)} 个{img_hint} [{st.session_state.retrieval_mode}]")
            elif st.session_state.use_web_search:
                st.info("🔄 No relevant documents found, falling back to web search...")

        # Step 3: Web search fallback
        if (
            (st.session_state.force_web_search or not context)
            and st.session_state.use_web_search
            and st.session_state.exa_api_key
        ):
            with st.spinner("🔍 Searching the web..."):
                try:
                    web_results, web_sources = run_web_search(rewritten_query, search_domains)
                    if web_results:
                        context = f"Web Search Results:\n{web_results}"
                        if st.session_state.force_web_search:
                            st.info("ℹ️ Using web search as requested via toggle.")
                        else:
                            st.info(
                                "ℹ️ Using web search as fallback since no relevant documents were found."
                            )
                except Exception as e:
                    st.error(f"❌ Web search error: {str(e)}")

        # Step 4: Generate response
        with st.spinner("🤖 Generating..."):
            try:
                # 获取当前轮次编号范围，用于 prompt 指令
                _cite_nums = list(st.session_state.cited_docs.keys()) if st.session_state.cited_docs else []
                _cite_hint = (
                    f"引用标号范围为 [{_cite_nums[0]}]~[{_cite_nums[-1]}]，" if _cite_nums else ""
                )
                if context:
                    full_prompt = f"""以下是检索到的参考文档，每段开头的 [编号] 是该文档块的引用标号：

{context}

原始问题：{prompt}
改写问题：{rewritten_query}

请根据以上参考文档，给出详细、准确的回答。
**重要要求**：在回答中，凡是引用了某文档块的信息，请在该句末尾用方括号标注引用编号，如 [1]、[2]、[3]。
{_cite_hint}如有多处引用同一文档，可重复标注。若某信息来自多个文档，请同时列出，如 [1][2]。
请用与用户问题相同的语言回答。"""
                else:
                    full_prompt = (
                        f"Original Question: {prompt}\nRewritten Question: {rewritten_query}"
                    )
                    st.info("ℹ️ No relevant information found in documents or web search.")

                if image_docs:
                    # 多模态路径：将 OSS 原图传给 qwen3.5-plus；表格块改用 md_table 作为文字上下文
                    _ctx_parts = []
                    for _d in docs:
                        _stype = _d.metadata.get("source_type")
                        if _stype == "image":
                            continue  # 图片走视觉通道，不加入文字上下文
                        _cn = next((k for k, v in st.session_state.cited_docs.items() if v is _d), "?")
                        if _stype == "excel":
                            _ctx_parts.append(
                                f"[{_cn}] [表格「{_d.metadata.get('file_name', '')}」- Sheet:「{_d.metadata.get('sheet_name', '')}」]\n"
                                f"{_d.metadata.get('md_table', _d.page_content)}"
                            )
                        else:
                            _ctx_parts.append(f"[{_cn}] {_d.page_content}")
                    text_only_ctx = "\n\n".join(_ctx_parts)
                    with st.status("🖼️ 调用 qwen3.5-plus 多模态模型处理图片上下文…", expanded=True) as mm_status:
                        img_names = "、".join([d.metadata.get("file_name", "未知") for d in image_docs])
                        st.write(f"📎 关联图片：{img_names}")
                        st.write("🔗 正在将 OSS 原图传入 qwen3.5-plus 模型…")
                        response = generate_multimodal_response(rewritten_query, image_docs, text_only_ctx, memory_context=memory_context)
                        mm_status.update(label="✅ 多模态回答生成完成", state="complete", expanded=False)
                else:
                    response = generate_rag_response(full_prompt, memory_context=memory_context)

                st.session_state.history.append(
                    {"role": "assistant", "content": response}
                )

                with st.chat_message("assistant"):
                    # 将回复中的 [N] 引用渲染为带颜色的徽章
                    import re as _re
                    def _render_cited_response(text: str) -> str:
                        """将 [N] 替换为醒目的内联引用徽章（HTML span）。"""
                        def _badge(m):
                            n = m.group(1)
                            return (
                                f'<span style="display:inline-block;background:#1565C0;color:#fff;'
                                f'border-radius:4px;padding:0 5px;font-size:0.78em;font-weight:600;'
                                f'margin:0 1px;vertical-align:middle;">[{n}]</span>'
                            )
                        return _re.sub(r'\[([0-9]+)\]', _badge, text)

                    rendered = _render_cited_response(response)
                    st.markdown(rendered, unsafe_allow_html=True)

                    # 来源卡片：每个引用块一个 expander
                    if not st.session_state.force_web_search and st.session_state.cited_docs:
                        st.markdown("---")
                        st.markdown("**📎 引用来源**")
                        for cite_num, doc in st.session_state.cited_docs.items():
                            source_type = doc.metadata.get("source_type", "unknown")
                            _icon = {"pdf": "📄", "excel": "📊", "image": "🖼️"}.get(source_type, "🌐")
                            with st.expander(f"**[{cite_num}]** {_icon} {_get_source_label(cite_num, doc)}", expanded=False):
                                _render_source_card(cite_num, doc)

                    if web_sources:
                        with st.expander("🌐 查看网络搜索来源"):
                            for i, src in enumerate(web_sources, 1):
                                st.markdown(f"**{i}. [{src['title']}]({src['url']})**")
                                content = src["content"]
                                preview = content if content else "No content extracted."
                                st.caption(preview)
                                st.divider()
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")

else:
    # st.warning("⚠️ Please enter your DashScope API Key to continue")
    st.warning("⚠️ 请输入Dashscope API Key以继续")
