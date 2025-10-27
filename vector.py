# import the required packages
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# read the data csv file
df = pd.read_csv("data.csv")

# Convert the 'mw' column from numeric to string (str)
df['mw'] = df['mw'].astype(str)
df['date'] = df['date'].astype(str)
df['time'] = df['time'].astype(str)

# choose the embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# specify the location to store the vector database
db_location = "./chroma_db"

# check if the database already exists if it does not exist we will add documents
add_documents = not os.path.exists(db_location)

if add_documents:
    # create a list to hold the documents
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Create readable text for embeddings
        text = (
            f"On {row['date']} at {row['time']}, "
            f"{row['fuel_type']} generated {row['mw']} megawatts of energy, "
            f"which was {row['fuel_percentage_of_total'] * 100:.2f}% of the total. "
            f"Renewable: {row['is_renewable']}."
        )

        # Store metadata (helps with filtering or retrieving source info later)
        metadata = {
            "date": row["date"],
            "time": row["time"],
            "fuel_type": row["fuel_type"],
            "is_renewable": bool(row["is_renewable"]),
            "fuel_percentage": float(row["fuel_percentage_of_total"]),
        }

        # Create the LangChain document
        document = Document(page_content=text, metadata=metadata)

        # Collect it
        documents.append(document)
        ids.append(str(i))

    print(f"✅ Created {len(documents)} Document objects.")

vector_store = Chroma(
    collection_name="energy_data",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 20}
)

'''
query = "Which renewable source generated the most power?"
results = vector_store.similarity_search(query, k=3)

for doc in results:
    print("📄", doc.page_content)
    print("🧾 Metadata:", doc.metadata)
    print("-" * 80)

'''