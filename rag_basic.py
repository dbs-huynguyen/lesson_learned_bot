from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


loader = DirectoryLoader(
    path="./data",
    glob="**/*.docx",
    loader_cls=UnstructuredFileLoader,
    show_progress=True,
    use_multithreading=True,
)
docs = loader.load()

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",  # Markdown headers
    "```\n",  # Code blocks
    "\n\\*\\*\\*+\n",  # Horizontal rules
    "\n---+\n",  # Horizontal rules
    "\n___+\n",  # Horizontal rules
    "\n\n",  # Paragraph breaks
    "\n",  # Line breaks
    " ",  # Spaces
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

splits = text_splitter.split_documents(docs)

embeddings = init_embeddings("huggingface:AITeamVN/Vietnamese_Embedding")

vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    normalize_L2=True,
    docstore=InMemoryDocstore(),
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5, "score_threshold": 0.2},
)

template = """You are a strict, citation-focused assistant for a private knowledge base.

RULES:
1) Use ONLY the provided context to answer.
2) If the answer is not clearly contained in the context, say: "I don't know based on the provided documents."
3) Do NOT use outside knowledge, guessing, or web information.
4) If applicable, cite sources as (source:page) using the metadata.

Context:
{context}

Question: {question}""".strip()

prompt = ChatPromptTemplate.from_template(template)

llm = init_chat_model(
    "ollama:qwen3.5:9b",
    base_url="http://192.168.88.179:11434",
    temperature=0,
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = input("Question: ")
answer = rag_chain.invoke(question)
print(answer)
