from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    RetrieverEvaluator,
)
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.llms import ChatMessage
import pandas as pd
import copy
import Stemmer
from typing import List
from llama_index.postprocessor.cohere_rerank import CohereRerank

def create_contextual_nodes(nodes_, llm_anthropic, prompt_document, prompt_chunk, WHOLE_DOCUMENT):
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