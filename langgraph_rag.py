import pprint

from langchain.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_docling.loader import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker


loader = DirectoryLoader(
    path="./data",
    glob="**/*.docx",
    loader_cls=UnstructuredFileLoader,
    show_progress=True,
    use_multithreading=True,
)
docs = loader.load()

for d in docs:
    pprint.pprint(f"{d.page_content}")
    print("-" * 80)
exit(0)
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

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=200,
    chunk_overlap=10,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1200,
#     chunk_overlap=200,
#     add_start_index=True,
#     strip_whitespace=True,
#     separators=MARKDOWN_SEPARATORS,
# )
splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b",
    base_url="http://192.168.88.179:11434",
)


vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="lessons_learned",
)

retriever = vectorstore.as_retriever(
    # search_type="similarity_score_threshold",
    # search_kwargs={
    #     "k": 5,
    #     "score_threshold": 0.5,
    # },
    search_type="mmr",
    search_kwargs={
        "k": 100,
        "fetch_k": 100,
        "lambda_mult": 0.7,
    },
)


@tool
def retrieve_lessons_learned(query: str) -> str:
    """Retrieve information on lessons learned."""
    docs = retriever.invoke(query)
    return "\n\n".join([f"{doc}" for doc in docs])


response_model = ChatOllama(
    model="qwen3.5:9b",
    reasoning=False,
    temperature=0,
    base_url="http://192.168.88.179:11434",
)


def generate_query_or_respond(state: MessagesState) -> dict[str, list[AIMessage]]:
    """Call the model to generate a response based on the current state. Give
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model.bind_tools([retrieve_lessons_learned]).invoke(
        state["messages"]
    )
    return {"messages": [response]}


# Grade edge
GRADE_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.

RULES:
1) If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
2) Give a binary_score 'yes' or 'no' score to indicate whether the document is relevant to the question.
3) Must ALWAYS follow the provided JSON Schema.

Here is the user question: {question}

Here is the retrieved document:
{context}""".strip()


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant."
    )


grader_model = ChatOllama(
    model="qwen3.5:9b",
    reasoning=False,
    temperature=0,
    num_predict=128,
    base_url="http://192.168.88.179:11434",
)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(context=context, question=question)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


# Rewrite node
REWRITE_PROMPT = """Look at the input and try to reason about the underlying semantic intent / meaning.

Here is the initial question:
 ------- 
{question}
 ------- 
Formulate an improved question:""".strip()


def rewrite_question(state: MessagesState) -> dict[str, list[HumanMessage]]:
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


# Answer node
GENERATE_PROMPT = """You are a strict, citation-focused assistant for a private knowledge base.

RULES:
1) Use ONLY the provided context to answer.
2) If the answer is not clearly contained in the context, say: "I don't know based on the provided documents."
3) Do NOT use outside knowledge, guessing, or web information.
4) If applicable, cite sources as file name, reference location using the metadata.

Question: {question}

Context:
{context}""".strip()


def generate_answer(state: MessagesState) -> dict[str, list[AIMessage]]:
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# Workflow
workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve_lessons_learned]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

# Define workflow
workflow.add_edge(START, "generate_query_or_respond")
# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END},
)
# Edges taken after the `action` node is called.
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()

while True:
    query = input("Question: ")
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": query}]}, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
