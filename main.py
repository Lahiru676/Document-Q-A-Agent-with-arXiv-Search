import os
import json
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from dotenv import load_dotenv
import arxiv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load  PDFs
PDF_DIR = "D:/AI/langgraph_doc_agent/documets" 
files = [
    os.path.join(PDF_DIR, f)
    for f in os.listdir(PDF_DIR)
    if f.lower().endswith(".pdf")
]
docs = []
for file in files:
    loader = PyMuPDFLoader(file)
    for page in loader.load():
        page.metadata["source_doc"] = file
        docs.append(page)
def load_and_prepare_documents(file_paths):
    all_docs = []
    for path in file_paths:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_doc"] = os.path.basename(path)
        all_docs.extend(docs)

    for doc in all_docs:
        text = doc.page_content.lower()
        if "abstract" in text[:300]:
            doc.metadata["section"] = "abstract"
        elif "introduction" in text:
            doc.metadata["section"] = "introduction"
        elif "method" in text:
            doc.metadata["section"] = "methodology"
        elif "result" in text:
            doc.metadata["section"] = "results"
        elif "conclusion" in text:
            doc.metadata["section"] = "conclusion"
        else:
            doc.metadata["section"] = "general"

    return all_docs

# Split and store
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./vectorstore",
    collection_name="papers"
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


#tools
@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for a paper based on a user-described topic or keywords."""
    search = arxiv.Search(query=query, max_results=1)
    for result in search.results():
        return f"**{result.title}**\n\n{result.summary}\n\n[Read more]({result.entry_id})"
    return "No relevant papers found on arXiv."

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from uploaded PDFs."""
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information"
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)

tools = [retriever_tool, search_arxiv]
llm_with_tools = llm.bind_tools(tools)



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent assistant that can answer questions using a document loaded into your knowledge base or by searching arXiv.
Use the retriever tool to find answers from the PDF.if the user ask for give equation use documents equations.
If the user asks for papers not in the PDF, use the arXiv search tool.
Always cite your sources.
"""

tools_dict = {tool.name: tool for tool in tools}

def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    message = llm_with_tools.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        if not t['name'] in tools_dict:
            result = "Incorrect Tool Name"
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {'messages': results}

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("llm", call_llm)
    workflow.add_node("retriever_agent", take_action)
    workflow.add_edge(START, "llm")
    workflow.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    workflow.add_edge("retriever_agent", "llm")
    return workflow.compile()

#save the chat history

def save_chat_history(messages, path="chat_history.json"):
    with open(path, "w") as f:
        json.dump([{"type": m.type, "content": m.content} for m in messages], f)

def load_chat_history(path="chat_history.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        raw = json.load(f)
    cls_map = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage, "tool": ToolMessage}
    return [cls_map[m["type"]](content=m["content"]) for m in raw]

# Main execution

if __name__ == "__main__":
    app = build_graph()
    chat_history = load_chat_history()

    user_input = "give me summary of optical flow"
    chat_history.append(HumanMessage(content=user_input))

    result = app.invoke({"messages": chat_history})
    reply = result["messages"][-1]

    print("RESPONSE:")
    print(reply.content)

    chat_history.append(reply)
    save_chat_history(chat_history)
