## reference: https://nvidia.github.io/GenerativeAIExamples/0.5.0/notebooks/04_llamaindex_hier_node_parser.html (too old to use)

# ── core imports ───────────────────────────────────────────────────────
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SummaryIndex

from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.schema import MetadataMode
from llama_index.core import SimpleDirectoryReader, Document

# ── global configuration ───────────────────────────────────────────────
# Set up global defaults
llm = OpenAI(model="gpt-4o-mini")
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

# ── ingest & chunk ────────────────────────────────────────────────────
docs = []
# we use the standard chartered bank's position statement as an example
for d in SimpleDirectoryReader(r"D:\\Downloads\\position_statement" ).load_data():
    d.metadata["deal_id"] = d.metadata["file_name"].split("_")[0]
    docs.append(d)


docs = [
            Document(text="\n\n".join(
                    document.get_content(metadata_mode=MetadataMode.ALL)
                    for document in docs
                )
            )
        ]


from llama_index.core.text_splitter import TokenTextSplitter
text_splitter_ids = ["1024", "510"]
text_splitter_map = {}
for ids in text_splitter_ids:
    text_splitter_map[ids] = TokenTextSplitter(
        chunk_size=int(ids),
        chunk_overlap=200
    )

large_chunk_size = 1536
node_parser = HierarchicalNodeParser.from_defaults(node_parser_ids=text_splitter_ids, node_parser_map=text_splitter_map)

nodes = node_parser.get_nodes_from_documents(docs)


# Configure hierarchical parser with levels and chunk sizes
# node_parser = HierarchicalNodeParser(
#     node_parser_map={
#         "high": SentenceSplitter(chunk_size=1024),
#         "mid": SentenceSplitter(chunk_size=512),
#         "low": SentenceSplitter(chunk_size=256),
#     },
#      levels=["high", "mid", "low"]  # ← Add this lin
# )
# node_parser = HierarchicalNodeParser.from_defaults(
#     chunk_size=256,
#     chunk_overlap=200
# )
# # Generate hierarchical nodes
# nodes = node_parser.get_nodes_from_documents(docs, include_extra_info=True)

# ── build indices ─────────────────────────────────────────────────────
vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
summary_index = SummaryIndex.from_documents(docs, llm=llm)

# ── retrieval engines ─────────────────────────────────────────────────
vec_retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=8,
    filters=[{"key": "deal_id", "value": "<placeholder>"}],
)

from llama_index.core.query_engine import RetrieverQueryEngine
deal_search = RetrieverQueryEngine.from_args(
    vec_retriever,
    node_postprocessors=[CohereRerank(top_n=4)],
    llm=llm
)

deal_search = RetrieverQueryEngine.from_args(
    vec_retriever,
    node_postprocessors=[CohereRerank(top_n=4)],
    llm=llm
)

# deal_search = vec_retriever.as_query_engine(
#     node_postprocessors=[CohereRerank(top_n=4)],
#     llm=llm
# )

portfolio_summary = summary_index.as_query_engine(
    response_mode="tree_summarize",
    llm=llm
)

# from llama_index.core.query_engine import RouterQueryEngine, QueryConditionSelector

# # Define routing logic
# def select_query_engine(query_str: str, metadata: dict) -> str:
#     return "deal" if "deal_id" in metadata else "portfolio"

# # Build router
# router = RouterQueryEngine(
#     selector=QueryConditionSelector(
#         query_engine_map={
#             "deal": deal_search,
#             "portfolio": portfolio_summary,
#         },
#         condition=select_query_engine,
#     )
# )

from llama_index.core.base.base_query_engine import BaseQueryEngine
from typing import Dict, Any

def custom_selector(metadata: Dict[str, Any]) -> str:
    """Select query engine based on metadata."""
    return "deal" if "deal_id" in metadata else "portfolio"

# Wrap router logic in a class or function
class CustomRouterQueryEngine(RouterQueryEngine):
    def __init__(
        self,
        deal_engine: BaseQueryEngine,
        portfolio_engine: BaseQueryEngine,
    ):
        self._deal_engine = deal_engine
        self._portfolio_engine = portfolio_engine

    def _get_query_engine(self, query_str: str, metadata: Dict[str, Any]) -> BaseQueryEngine:
        selected = custom_selector(metadata)
        return self._deal_engine if selected == "deal" else self._portfolio_engine

    def query(self, query_str: str, **kwargs) -> Any:
        metadata = kwargs.get("metadata", {})
        selected_engine = self._get_query_engine(query_str, metadata)
        return selected_engine.query(query_str, **kwargs)

    async def aquery(self, query_str: str, **kwargs) -> Any:
        metadata = kwargs.get("metadata", {})
        selected_engine = self._get_query_engine(query_str, metadata)
        return await selected_engine.aquery(query_str, **kwargs)

router = CustomRouterQueryEngine(
    deal_engine=deal_search,
    portfolio_engine=portfolio_summary,
)

# ── examples ───────────────────────────────────────────────────────────
print(router.query("Summarise every industry position statement"))
print(router.query("Summarise nature industry position statement"))
print(router.query("compare difference between nature industry and climate change position statement"))
