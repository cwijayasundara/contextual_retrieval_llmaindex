import nest_asyncio
import warnings
import pandas as pd
import copy
import Stemmer
import os
from dotenv import load_dotenv
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    RetrieverEvaluator,
)
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.llms import ChatMessage
from typing import List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.llms.openai import OpenAI

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

llm_anthropic = Anthropic(model="claude-3-5-sonnet-20240620")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

documents = SimpleDirectoryReader(
    input_files=["docs/paul_graham_essay.txt"],
).load_data()

WHOLE_DOCUMENT = documents[0].text

prompt_document = """<document>
{WHOLE_DOCUMENT}
</document>"""

prompt_chunk = """Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving 
search retrieval of the chunk. Answer only with the succinct context and nothing else."""

# utility functions
def create_contextual_nodes(nodes_):
    """Function to create contextual nodes for a list of nodes"""
    nodes_modified = []
    for node in nodes_:
        new_node = copy.deepcopy(node)
        messages = [
            ChatMessage(role="system", content="You are helpful AI Assistant."),
            ChatMessage(
                role="user",
                content=[
                    {
                        "text": prompt_document.format(
                            WHOLE_DOCUMENT=WHOLE_DOCUMENT
                        ),
                        "type": "text",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "text": prompt_chunk.format(CHUNK_CONTENT=node.text),
                        "type": "text",
                    },
                ],
            ),
        ]
        new_node.metadata["context"] = str(
            llm_anthropic.chat(
                messages,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
        )
        nodes_modified.append(new_node)
    return nodes_modified

def create_embedding_retriever(nodes_, similarity_top_k=2):
    """Function to create an embedding retriever for a list of nodes"""
    vector_index = VectorStoreIndex(nodes_)
    retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
    return retriever

def create_bm25_retriever(nodes_, similarity_top_k=2):
    """Function to create a bm25 retriever for a list of nodes"""
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes_,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    return bm25_retriever

def create_eval_dataset(nodes_, llm, num_questions_per_chunk=2):
    """Function to create a evaluation dataset for a list of nodes"""
    qa_dataset = generate_question_context_pairs(
        nodes_, llm=llm, num_questions_per_chunk=num_questions_per_chunk
    )
    return qa_dataset

def set_node_ids(nodes_):
    """Function to set node ids for a list of nodes"""
    # by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
    for index, node in enumerate(nodes_):
        node.id_ = f"node_{index}"
    return nodes_

async def retrieval_results(retriever, eval_dataset):
    """Function to get retrieval results for a retriever and evaluation dataset"""
    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        metrics, retriever=retriever
    )
    eval_results = await retriever_evaluator.aevaluate_dataset(eval_dataset)
    return eval_results

def display_results(name, eval_results):
    """Display results from evaluate."""
    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)
    full_df = pd.DataFrame(metric_dicts)
    columns = {
        "retrievers": [name],
        **{k: [full_df[k].mean()] for k in metrics},
    }
    metric_df = pd.DataFrame(columns)
    return metric_df

class EmbeddingBM25RerankerRetriever(BaseRetriever):
    """Custom retriever that uses both embedding and bm25 retrievers and reranker"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        reranker: CohereRerank,
    ) -> None:
        """Init params."""
        self._vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        vector_nodes.extend(bm25_nodes)
        retrieved_nodes = self.reranker.postprocess_nodes(
            vector_nodes, query_bundle
        )
        return retrieved_nodes

# create nodes
node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)
# set node ids
nodes = set_node_ids(nodes)
# create contextual nodes
nodes_contextual = create_contextual_nodes(nodes)
print(nodes[0].metadata, nodes_contextual[0].metadata)

# set similarity top k
similarity_top_k = 3

# Set CohereReranker
cohere_rerank = CohereRerank(
    api_key=os.environ["COHERE_API_KEY"], top_n=similarity_top_k
)

# Create retrievers
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

# Create Synthetic query dataset
llm = OpenAI(model="gpt-4o-2024-08-06")
qa_dataset = create_eval_dataset(nodes, llm=llm, num_questions_per_chunk=2)
print(list(qa_dataset.queries.values())[1])

async def embedding_retriever_results_fn():
    # Evaluate retrievers with and without contextual nodes
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
        contextual_embedding_bm25_retriever_rerank, qa_dataset
    )
    return contextual_embedding_bm25_retriever_rerank_results

# Without Context
print("without context", pd.concat(
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
)

# with Context
print("with context", pd.concat(
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
))
