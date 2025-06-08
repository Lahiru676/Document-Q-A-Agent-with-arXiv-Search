# Document-Q-A-Agent-with-arXiv-Search
This agent can read pdf and answer the questions and if there no any relevant document for questions it can check arxiv search and answer the question  
![image](https://github.com/user-attachments/assets/ef0a0177-e85f-42a6-83af-327799b5e4c7)
![image](https://github.com/user-attachments/assets/aa89df8f-81c2-465a-8a01-6d6cdb4b6b66)


# 📄 LangGraph Document Q&A + ArXiv Search Agent

This project is an **enterprise-ready AI agent** for answering questions from **academic PDFs** using **LangGraph**, **OpenAI GPT-4o**, and **arXiv** integration. It features persistent **chat history**, automatic **multi-PDF ingestion**, and an intuitive **Streamlit UI** for interactive querying.

---

##  Features

### ✅ Document Q&A Agent
- Uses `LangGraph` to orchestrate intelligent conversations.
- Connects with `OpenAI GPT-4o` for accurate natural language answers.
- Queries relevant PDF content using **vector search** with `ChromaDB`.

### ✅ Multi-PDF Ingestion + Metadata Extraction
- Automatically loads all PDFs from the `documents/` folder.
- Extracts and tags sections like `Abstract`, `Methodology`, `Conclusion`, etc.
- Splits text for high-accuracy retrieval using `RecursiveCharacterTextSplitter`.

### ✅ ArXiv Functional Tool
- If your question references external research, the AI will **search arXiv.org** using keyword-based queries via the `arxiv` Python package.

### ✅ Persistent Chat History
- Saves chat history per session in `chat_history_<session_id>.json`.
- Fully replayable and editable for audit or review.

### ✅ Streamlit Frontend
- Clean UI for chatting with the agent.
- Session ID support for **multi-user** behavior.
- “🧹 Clear History” button to reset the chat.
- Future extensibility for file upload or voice input.

---

## 🛠️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/Lahiru676/Document-Q-A-Agent-with-arXiv-Search.git
cd Document-Q-A-Agent-with-arXiv-Search
2. Install Dependencies
Using pip or conda environment:
pip install -r requirements.txt

✅ Ensure your .env file contains your OpenAI key:

OPENAI_API_KEY=your-key-here
3. Add PDF Documents
Place your academic or technical PDFs inside:
/documents/
4. Run the Streamlit App
streamlit run app.py
💡 Example Queries
"Summarize the methodology of the optical flow paper."

"What are the results discussed in the fiber tracking paper?"

"Find a recent paper on convolutional transformers from arXiv."

"Compare accuracy metrics reported in the PDFs."

📁 Project Structure

Document-Q-A-Agent-with-arXiv-Search/
│
├── main.py              # LangGraph-powered LLM agent
├── app.py               # Streamlit UI
├── documents/           # Folder for input PDFs
├── vectorstore/         # Chroma vector store (auto-generated)
├── chat_history_*.json  # User chat logs (auto-generated)
├── requirements.txt     # Python dependencies
├── .env                 # OpenAI API key
└── README.md


🧠 Tech Stack
LangGraph (LLM Orchestration)
LangChain
ChromaDB (Vector Store)
Streamlit (Frontend)
OpenAI GPT-4o
arXiv API


