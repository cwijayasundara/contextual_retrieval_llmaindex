import warnings
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

warnings.filterwarnings('ignore')
_ = load_dotenv()

Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

documents = SimpleDirectoryReader("docs").load_data()

splitter = SentenceSplitter(chunk_size=512)

nodes = splitter.get_nodes_from_documents(documents)

# BM25 Retriever + Disk Persistence
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=2,
    # Optional: We can pass in the stemmer and set the language for stopwords
    # This is important for removing stopwords and stemming the query + text
    # The default is english for both
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)

bm25_retriever.persist("./bm25_retriever")

loaded_bm25_retriever = BM25Retriever.from_persist_dir("./bm25_retriever")

Query = "What did the author do after RISD?"
retrieved_nodes = loaded_bm25_retriever.retrieve(Query)

# plain BM25 and with Persistence
# for node in retrieved_nodes:
#     print(node.text)


# BM25 Retriever + Docstore Persistence

# initialize a docstore to store nodes
# also available are mongodb, redis, postgres, etc for docstores

from llama_index.core.storage.docstore import SimpleDocumentStore

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

# We can pass in the index, docstore, or list of nodes to create the retriever
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=2,
    # Optional: We can pass in the stemmer and set the language for stopwords
    # This is important for removing stopwords and stemming the query + text
    # The default is english for both
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)

from llama_index.core.response.notebook_utils import display_source_node

# will retrieve context from specific companies
retrieved_nodes = bm25_retriever.retrieve( "What happened at Viaweb and Interleaf?")

# for node in retrieved_nodes:
#     print(node.text)

# Hybrid Retriever with BM25 + Chroma

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("dense_vectors")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(
    docstore=docstore, vector_store=vector_store
)

index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

import nest_asyncio

nest_asyncio.apply()

from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [
        index.as_retriever(similarity_top_k=2),
        BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=2
        ),
    ],
    num_queries=1,
    use_async=True,
)

nodes = retriever.retrieve("What happened at Viaweb and Interleaf?")
for node in nodes:
    print(node.text)