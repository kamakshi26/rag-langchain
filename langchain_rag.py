from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ─── STEP 1: Load PDF ────────────────────────────────────────
print("Step 1: Loading PDF...")
loader = PyPDFLoader("p15.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages\n")

# ─── STEP 2: Chunk ───────────────────────────────────────────
print("Step 2: Chunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks\n")

# ─── STEP 3+4: Embed + Store ─────────────────────────────────
print("Step 3+4: Embedding and storing in Qdrant...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="irs_pub15_langchain",
    force_recreate=True
)
print("Stored in Qdrant\n")

# ─── STEP 5+6: Retrieve + Generate ───────────────────────────
print("Step 5+6: Setting up retrieval chain...")

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a tax advisor. Answer using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided document."
Always end with: Source: [quote the exact sentence you used]

Context: {context}

Question: {question}
Answer:"""
)

llm = Ollama(model="llama3.2", temperature=0)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": custom_prompt}
)

# ─── TEST ─────────────────────────────────────────────────────
def ask(question):
    print(f"Question: {question}")
    result = chain.invoke({"query": question})
    print(f"Answer: {result['result']}")
    print("\n" + "="*60 + "\n")

ask("What is the social security tax rate for 2026?")
ask("What is the FUTA tax rate?")
ask("What happens if I deposit taxes late?")
ask("What is the capital of France?")