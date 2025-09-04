import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import tempfile
import os

INDEX_PATH = "faiss_index"

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
        سناریوی تست نمونه:
        شرح: {example_desc}
        مراحل:
        {example_ts}
        """

    base_prompt = f"""
    شما یک تحلیل‌گر سیستم هستید. یک سناریوی تست گام‌به‌گام برای بررسی نیازمندی زیر به زبان فارسی تولید کنید.
    نیازمندی‌ها:
    {requirements}
    توضیح سناریوی مورد نظر:
    {ts_description}
    {example_part}
    در نهایت، لطفاً یک جمله بنویسید که چرا این سناریو تست نیازمندی بالا را پوشش می‌دهد.
    """
    return base_prompt

# --- 3. Generate test scenario ---
def generate_test_scenario(prompt, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(prompt)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = f"""
    متن‌های بازیابی‌شده:
    {context}

    پرسش:
    {prompt}
    """

    llm = ChatOpenAI(
        model='gpt-4o',
        api_key="FAKE",
        base_url="http://localhost:1050/v1",
        temperature=0
    )

    try:
        return llm.invoke(final_prompt)
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
