## older version of llama_index

import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import nest_asyncio

nest_asyncio.apply()

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")

# from trulens_eval import (
#     Feedback,
#     TruLlama,
#     OpenAI
# )

# from trulens_eval.feedback import Groundedness

# def get_prebuilt_trulens_recorder(query_engine, app_id):
#     openai = OpenAI()

#     qa_relevance = (
#         Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
#         .on_input_output()
#     )

#     qs_relevance = (
#         Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
#         .on_input()
#         .on(TruLlama.select_source_nodes().node.text)
#         .aggregate(np.mean)
#     )

#     grounded = Groundedness(groundedness_provider=openai)

#     groundedness = (
#         Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
#             .on(TruLlama.select_source_nodes().node.text)
#             .on_output()
#             .aggregate(grounded.grounded_statements_aggregator)
#     )

#     feedbacks = [qa_relevance, qs_relevance, groundedness]
#     tru_recorder = TruLlama(
#         query_engine,
#         app_id=app_id,
#         feedbacks=feedbacks
#     )
#     return tru_recorder

from importlib.metadata import version
from packaging.version import parse
if parse(version('llama_index')) <parse('0.10'):
    from llama_index import ServiceContext, VectorStoreIndex, StorageContext
    from llama_index.node_parser import SentenceWindowNodeParser
    from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
    from llama_index.indices.postprocessor import SentenceTransformerRerank
    from llama_index import load_index_from_storage
else:
    from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
    from llama_index.core.node_parser import SentenceWindowNodeParser
    from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
    from llama_index.core.indices.postprocessor import SentenceTransformerRerank
    from llama_index.core import load_index_from_storage

import os


def build_sentence_window_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=5,
    save_dir="sentence_index",
    rebuild=True
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if (not os.path.exists(save_dir)) or rebuild:
        print('building the vector indexing')
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        print('loading the vector indexing')
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )
    print(sentence_index)

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
    verbose=True
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    print('building the sentence_window_engine')
    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, 
        node_postprocessors=[postproc, rerank],
        verbose=verbose,
    )
    return sentence_window_engine

if parse(version('llama_index')) <parse('0.10'):
    from llama_index.node_parser import HierarchicalNodeParser
    from llama_index.node_parser import get_leaf_nodes
    from llama_index import StorageContext
    from llama_index.retrievers import AutoMergingRetriever
    from llama_index.indices.postprocessor import SentenceTransformerRerank
    from llama_index.query_engine import RetrieverQueryEngine
else:
    from llama_index.core.node_parser import HierarchicalNodeParser
    from llama_index.core.node_parser import get_leaf_nodes
    from llama_index.core import StorageContext
    from llama_index.core.retrievers import AutoMergingRetriever
    from llama_index.core.indices.postprocessor import SentenceTransformerRerank
    from llama_index.core.query_engine import RetrieverQueryEngine    

def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
    rebuild=True
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if (not os.path.exists(save_dir)) or rebuild:
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine



# Function to mix keyword search and semantic search
def mixed_retrieval(index, query, keyword_weight=0.5, semantic_weight=0.5):
    # Perform keyword search
    keyword_results = index.search(query, method='keyword')
    
    # Perform semantic search
    semantic_results = index.search(query, method='semantic')
    
    # Combine results
    combined_results = {}
    
    for result in keyword_results:
        doc_id = result['doc_id']
        score = result['score'] * keyword_weight
        if doc_id in combined_results:
            combined_results[doc_id] += score
        else:
            combined_results[doc_id] = score
    
    for result in semantic_results:
        doc_id = result['doc_id']
        score = result['score'] * semantic_weight
        if doc_id in combined_results:
            combined_results[doc_id] += score
        else:
            combined_results[doc_id] = score
    
    # Sort results by combined score
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results