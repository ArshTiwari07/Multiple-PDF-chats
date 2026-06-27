# 📚 Chat with Multiple PDFs

A Streamlit app that lets you upload multiple PDF documents and have a conversational Q&A with their content — powered by **Groq** (LLaMA 3) for fast LLM inference and **HuggingFace** embeddings for free, local vector search.

---

## 🚀 Features

- Upload one or multiple PDF files
- Extracts and indexes all text automatically
- Ask questions in natural language
- Maintains conversation history across turns
- Fully free to run — no OpenAI key required

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyPDF2 |
| Text Splitting | LangChain Text Splitters |
| Embeddings | HuggingFace `hkunlp/instructor-xl` |
| Vector Store | FAISS |
| LLM | Groq (`llama3-8b-8192`) |
| Memory | LangChain ConversationBufferMemory |

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/multi-pdf-chat.git
cd multi-pdf-chat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ `torch` and `sentence-transformers` are large downloads (~2GB). Make sure you have a stable internet connection.

### 3. Get a Groq API key

Sign up for free at [console.groq.com](https://console.groq.com) and copy your API key.

### 4. Create a `.env` file

Create a file named `.env` in the project root and add:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 📖 How to Use

1. Launch the app — it opens in your browser at `http://localhost:8501`
2. In the **sidebar**, upload one or more PDF files
3. Click **"Process"** and wait for the documents to be indexed
4. Type any question in the chat input and hit Enter
5. The app answers based on the content of your PDFs and remembers the conversation

---

## 📁 Project Structure

```
multi-pdf-chat/
├── app.py              # Main application
├── htmlTemplates.py    # HTML/CSS templates for chat bubbles
├── requirements.txt    # Python dependencies
├── .env                # API keys (do not commit this)
└── README.md
```

---

## 🔄 Switching LLM Models

In `app.py`, inside `get_conversation_chain()`, change `model_name` to any Groq-supported model:

```python
llm = ChatGroq(
    model_name="llama3-8b-8192",       # fast & lightweight
    # model_name="llama3-70b-8192",    # more powerful
    # model_name="mixtral-8x7b-32768", # large context window
    # model_name="gemma-7b-it",        # Google Gemma
    temperature=0,
)
```

---

## 🔒 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com |

Never commit your `.env` file. Add it to `.gitignore`:

```
.env
```

---

## 📄 License

MIT License. Free to use and modify.
