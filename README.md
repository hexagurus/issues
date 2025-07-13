

```python
import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd

# 1. Load .env and your key
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY in .env")

# 2. Init embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    api_key=api_key
)

# 3. Load & split documents
books = pd.read_csv("books_cleaned.csv")
books["tagged_description"].to_csv("tagged_description.txt", sep="\n", index=False, header=False)
raw_docs = TextLoader("tagged_description.txt").load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
documents = splitter.split_documents(raw_docs)

# 4. Create an *empty* Chroma store
db = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_books"
)

# 5. Helper to chunk a list into batches
def chunk_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

# 6. Upsert in batches *with* delays
batch_size = 20          # combine 20 docs per request
delay_seconds = 1.0      # 1 second between batches

for batch in chunk_list(documents, batch_size):
    # 6a. Extract texts and IDs in one go
    texts = [doc.page_content for doc in batch]
    meta  = [doc.metadata      for doc in batch]  # if you have metadata to store

    # 6b. Embed & upsert in a single API call
    db.add_texts(texts, metadatas=meta)

    # 6c. Short delay to stay under rate limits
    time.sleep(delay_seconds)

# 7. Persist to disk
db.persist()

# 8. Now you can run your similarity search
results = db.similarity_search("A book to teach children about nature", k=10)
print(results)
```

**Why this helps**

* **Batching**: by embedding 20 chunks at once you reduce the total number of HTTP calls, which Google’s API prefers (fewer, larger requests).
* **Delays**: after each batch we `sleep(1)` sec, which spaces out calls so you don’t burst above your “requests per minute” quota.

