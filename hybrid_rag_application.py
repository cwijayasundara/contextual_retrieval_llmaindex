import warnings
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import nest_asyncio
from llama_index.core.retrievers import QueryFusionRetriever

warnings.filterwarnings('ignore')
_ = load_dotenv()

Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

documents = SimpleDirectoryReader("docs/policy").load_data()
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)

# initialize a docstore to store nodes
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

# We can pass in the index, docstore, or list of nodes to create the retriever
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=2,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)

bm25_retriever.persist("./bm25_retriever")
loaded_bm25_retriever = BM25Retriever.from_persist_dir("./bm25_retriever")

# Hybrid Retriever with BM25 + Chroma

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("dense_vectors")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(
    docstore=docstore, vector_store=vector_store
)

index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

nest_asyncio.apply()

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

nodes = retriever.retrieve("What is the cashback option for dentist fees?")
for node in nodes:
    print(node.text)