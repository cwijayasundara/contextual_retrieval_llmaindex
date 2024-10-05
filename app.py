from dotenv import load_dotenv
import warnings
import nest_asyncio
import os
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from util import (set_node_ids, create_contextual_nodes, create_embedding_retriever, create_bm25_retriever,
                  EmbeddingBM25RerankerRetriever, create_eval_dataset, retrieval_results, display_results)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.llms.openai import OpenAI
import pandas as pd

warnings.filterwarnings('ignore')
_ = load_dotenv()

nest_asyncio.apply()

llm_anthropic = Anthropic(model="claude-3-5-sonnet-20240620")

llm = OpenAI(model="gpt-4o")

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

documents = SimpleDirectoryReader(
    # input_files=["./paul_graham_essay.txt"],
    input_files=["docs/paul_graham_essay.txt"],
).load_data()

# extract the text from the document
WHOLE_DOCUMENT = documents[0].text

## Prompts for creating context for each chunk
prompt_document = """<document>
{WHOLE_DOCUMENT}
</document>"""

prompt_chunk = """Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving 
search retrieval of the chunk. Answer only with the succinct context and nothing else."""


node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

# set node ids
nodes = set_node_ids(nodes)

# Create contextual nodes : takes a while
nodes_contextual = create_contextual_nodes(nodes, llm_anthropic, prompt_document, prompt_chunk, WHOLE_DOCUMENT)

similarity_top_k = 3

cohere_rerank = CohereRerank(
    api_key=os.environ["COHERE_API_KEY"], top_n=similarity_top_k
)

# Create retrievers.
embedding_retriever = create_embedding_retriever(
    nodes, similarity_top_k=similarity_top_k
)
bm25_retriever = create_bm25_retriever(
    nodes, similarity_top_k=similarity_top_k
)
embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    embedding_retriever, bm25_retriever, reranker=cohere_rerank
)

# Create retrievers using contextual nodes.
contextual_embedding_retriever = create_embedding_retriever(
    nodes_contextual, similarity_top_k=similarity_top_k
)
contextual_bm25_retriever = create_bm25_retriever(
    nodes_contextual, similarity_top_k=similarity_top_k
)
contextual_embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    contextual_embedding_retriever,
    contextual_bm25_retriever,
    reranker=cohere_rerank,
)


qa_dataset = create_eval_dataset(nodes, llm=llm, num_questions_per_chunk=2)

## Evaluate retrievers with and without nodes
async def embedding_retriever_results_fn():
    embedding_retriever_results = await retrieval_results(
        embedding_retriever, qa_dataset
    )
    return embedding_retriever_results

async def bm25_retriever_results_fn():
    bm25_retriever_results = await retrieval_results(bm25_retriever, qa_dataset)
    return bm25_retriever_results

async def embedding_bm25_retriever_rerank_results_fn():
    embedding_bm25_retriever_rerank_results = await retrieval_results(
        embedding_bm25_retriever_rerank, qa_dataset
    )
    return embedding_bm25_retriever_rerank_results

# display results without contextual nodes
pd.concat(
    [
        display_results("Embedding Retriever", embedding_retriever_results_fn),
        display_results("BM25 Retriever", bm25_retriever_results_fn),
        display_results(
            "Embedding + BM25 Retriever + Reranker",
            embedding_bm25_retriever_rerank_results_fn,
        ),
    ],
    ignore_index=True,
    axis=0,
)

# Evaluate retrievers with contextual nodes

async def contextual_embedding_retriever_results_fn():
    contextual_embedding_retriever_results = await retrieval_results(
        contextual_embedding_retriever, qa_dataset
    )
    return contextual_embedding_retriever_results

async def contextual_bm25_retriever_results_fn():
    contextual_bm25_retriever_results = await retrieval_results(
        contextual_bm25_retriever, qa_dataset
    )
    return contextual_bm25_retriever_results

async def contextual_embedding_bm25_retriever_rerank_results_fn():
    contextual_embedding_bm25_retriever_rerank_results = await retrieval_results(
        contextual_embedding_bm25_retriever_rerank, qa_dataset)
    return contextual_embedding_bm25_retriever_rerank_results

# display results with contextual nodes

pd.concat(
    [
        display_results(
            "Contextual Embedding Retriever",
            contextual_embedding_retriever_results_fn,
        ),
        display_results(
            "Contextual BM25 Retriever", contextual_bm25_retriever_results_fn
        ),
        display_results(
            "Contextual Embedding + Contextual BM25 Retriever + Reranker",
            contextual_embedding_bm25_retriever_rerank_results_fn,
        ),
    ],
    ignore_index=True,
    axis=0,
)