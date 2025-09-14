import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import tempfile
from dotenv import load_dotenv
import os

load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
HF_MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", 512))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", 0.0))

# --- 1. Index documents ---
def index_documents(pdf_files):
    docs = []
    for uploaded_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            docs += loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = FAISS.from_documents(chunks, embedder)
    vectordb.save_local(INDEX_PATH)
    return vectordb, embedder

# --- 2. Create prompt ---
def create_prompt(requirements, ts_description, example_ts=None, example_desc=None):
    example_part = ""
    if example_ts and example_desc:
        example_part = f"""
            نمونه توضیح یک سناریوی تست:
        {example_ts}
        سناریوی تست نمونه با تمام مراحل:
        {example_desc}
        """

    base_prompt = f"""
    شما یک تحلیل‌گر سیستم هستید و وظیفه دارید یک سناریوی تست دقیق برای پروژه مورد نظر تولید کنید. 
    هدف این سناریو بررسی عملکرد سیستم بر اساس نیازمندی‌ها و توضیحات ارائه‌شده است. 
    لطفاً یک سناریوی تست با توضیح گام‌به‌گام تولید کنید که نشان دهد چگونه عملکرد زیر آزمایش می‌شود:
    توضیح سناریوی مورد نظر:
    {ts_description}
    نیازمندی‌های پروژه:
    {requirements}
    {example_part}
    در خروجی نهایی، علاوه بر سناریوی تست با مراحل کامل، یک جمله یا دو جمله توضیح دهید که این سناریو چگونه نیازمندی‌ها و توضیح سناریوی مورد نظر را پوشش می‌دهد.
    """
    return base_prompt

# --- 3. Generate test scenario ---
def generate_test_scenario(prompt, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke(prompt)
    context = "\n\n".join([d.page_content for d in docs])
    final_prompt = f"""
    context:
    {context}

    question:
    {prompt}
    """

    llm_endpoint = HuggingFaceEndpoint(
        repo_id=HF_MODEL_REPO,
        task="text2text-generation",
        huggingfacehub_api_token=HF_API_TOKEN,
        temperature=HF_TEMPERATURE,
        max_new_tokens=HF_MAX_TOKENS
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)

    try:
        return chat_model.invoke(final_prompt)
    except Exception as e:
        return f"خطا در ارتباط با مدل: {e}"
    
# --- 4. Streamlit UI ---
st.set_page_config(page_title="تولید سناریوی تست", layout="wide")
st.markdown(
    """
    <style>
    html, body, .main, [data-testid="stAppViewContainer"] {
        direction: rtl !important;
        text-align: right !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧪Test Scenario Generation")

uploaded_pdfs = st.file_uploader(
    "فایل‌های PDF دامنه را انتخاب کنید",
    type="pdf",
    accept_multiple_files=True
)

if st.button("بارگذاری/ایندکس PDF"):
    if uploaded_pdfs:
        with st.spinner("در حال پردازش PDFها..."):
            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            vectordb, embedder = index_documents(uploaded_pdfs)
            st.success("ایندکس جدید ساخته و ذخیره شد.")
    else:
        st.warning("لطفاً حداقل یک PDF بارگذاری کنید.")


requirements = st.text_area("نیازمندی‌ها")
description = st.text_area("توضیح سناریوی مورد نظر")
example_desc = st.text_area("شرح سناریوی نمونه (اختیاری)")
example_steps = st.text_area("مراحل سناریوی نمونه (اختیاری)")



if st.button("تولید سناریو"):
    if not os.path.exists(INDEX_PATH):
        st.warning("ابتدا PDFها را بارگذاری/ایندکس کنید.")
    elif not requirements or not description:
        st.warning("لطفاً نیازمندی‌ها و توضیح سناریو را وارد کنید.")
    else:
        with st.spinner("در حال تولید سناریو..."):
            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            vectordb = FAISS.load_local(INDEX_PATH, embedder, allow_dangerous_deserialization=True)
            prompt = create_prompt(requirements, description, example_steps, example_desc)
            result = generate_test_scenario(prompt, vectordb)
        st.subheader("سناریوی تولید شده:")
        st.markdown(result.content)
