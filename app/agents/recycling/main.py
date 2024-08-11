from typing import Optional
import pickle
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple, List
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from typing_extensions import Self
from app.dependencies import client
from app.agents.recycling.nodes import (
    is_image_present,
    identify_image_items,
    rephrase_question_based_on_image_items,
    rephrase_question_based_on_chat_history,
    retrieve_docs,
    grade_retrieved_docs,
    perform_web_search,
    generate_answer,
    generate_answer_from_llm,
    persist_chat_messages,
    should_do_web_search,
    State,
)

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter


# region: checkpoint saver
class AsyncFirestoreDBSaver(BaseCheckpointSaver):
    """A checkpoint saver that stores checkpoints in a Firestore database asynchronously."""

    client: firestore.AsyncClient
    checkpoints_collection: firestore.AsyncCollectionReference
    checkpoint_writes_collection: firestore.AsyncCollectionReference
    batch: firestore.AsyncWriteBatch

    def __init__(
        self,
        client: firestore.AsyncClient,
        checkpoints_collection_name: str,
        checkpoint_writes_collection_name: str,
    ) -> None:
        super().__init__()
        self.client = client
        self.checkpoints_collection = self.client.collection(
            checkpoints_collection_name
        )
        self.checkpoint_writes_collection = self.client.collection(
            checkpoint_writes_collection_name
        )
        self.batch = self.client.batch()

    @classmethod
    @asynccontextmanager
    async def from_conn_info(
        cls,
        *,
        checkpoints_collection_name: str,
        checkpoint_writes_collection_name: str,
    ) -> AsyncIterator["AsyncFirestoreDBSaver"]:
        client = None
        try:
            client = firestore.AsyncClient()
            yield AsyncFirestoreDBSaver(
                client, checkpoints_collection_name, checkpoint_writes_collection_name
            )
        finally:
            if client:
                client.close()

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the Firestore database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        query = self.checkpoints_collection.where(
            filter=FieldFilter("thread_id", "==", thread_id)
        ).where(filter=FieldFilter("checkpoint_ns", "==", checkpoint_ns))
        if checkpoint_id := get_checkpoint_id(config):
            query = query.where(
                filter=FieldFilter("checkpoint_id", "==", checkpoint_id)
            )
        result = (
            query.order_by("checkpoint_id", direction="DESCENDING").limit(1).stream()
        )
        async for doc in result:
            doc = doc.to_dict()
            checkpoint_id = doc["checkpoint_id"]
            config_values = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            serialized_writes = (
                self.checkpoint_writes_collection.where(
                    filter=FieldFilter("thread_id", "==", thread_id)
                )
                .where(filter=FieldFilter("checkpoint_ns", "==", checkpoint_ns))
                .where(filter=FieldFilter("checkpoint_id", "==", checkpoint_id))
                .stream()
            )
            pending_writes = [
                (
                    doc["task_id"],
                    doc["channel"],
                    self.serde.loads_typed((doc["type"], doc["value"])),
                )
                async for doc in serialized_writes
            ]
            return CheckpointTuple(
                {"configurable": config_values},
                checkpoint,
                self.serde.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes,
            )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the Firestore database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        query = self.checkpoints_collection
        if config is not None:
            query = query.where(
                filter=FieldFilter(
                    "thread_id", "==", config["configurable"]["thread_id"]
                )
            )
            if config["configurable"].get("checkpoint_ns", None):
                checkpoint_ns = config["configurable"].get("checkpoint_ns")
                query = query.where(
                    filter=FieldFilter("checkpoint_ns", "==", checkpoint_ns)
                )

        if filter:
            for key, value in filter.items():
                query = query.where(filter=FieldFilter(f"metadata.{key}", "==", value))

        if before is not None:
            query = query.where(
                filter=FieldFilter(
                    "checkpoint_id", "<", before["configurable"]["checkpoint_id"]
                )
            )

        result = query.order_by("checkpoint_id", direction="DESCENDING")

        if limit is not None:
            result = result.limit(limit)
        async for doc in result.stream():
            doc = doc.to_dict()
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["checkpoint_id"],
                    }
                },
                checkpoint,
                self.serde.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "checkpoint_ns": doc["checkpoint_ns"],
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the Firestore database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        doc = {
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized_checkpoint,
            "metadata": self.serde.dumps(metadata),
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }

        # Perform your operations here
        # Upsert in firestore
        # Query to find the document
        query = (
            self.checkpoints_collection.where(
                filter=FieldFilter("thread_id", "==", thread_id)
            )
            .where(filter=FieldFilter("checkpoint_ns", "==", checkpoint_ns))
            .where(filter=FieldFilter("checkpoint_id", "==", checkpoint_id))
        )

        # Get the document snapshot
        docs = await query.limit(1).get()

        if docs:
            # Document exists, update it
            doc_ref: firestore.AsyncDocumentReference = docs[0].reference
            await doc_ref.set(doc, merge=True)
        else:
            # Document does not exist, create a new one
            await self.checkpoints_collection.add(doc)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            type_, serialized_value = self.serde.dumps_typed(value)
            query = (
                self.checkpoint_writes_collection.where(
                    filter=FieldFilter("thread_id", "==", thread_id)
                )
                .where(filter=FieldFilter("checkpoint_ns", "==", checkpoint_ns))
                .where(filter=FieldFilter("checkpoint_id", "==", checkpoint_id))
                .where(filter=FieldFilter("task_id", "==", task_id))
                .where(filter=FieldFilter("idx", "==", idx))
            )
            docs: List[firestore.DocumentSnapshot] = await query.limit(1).get()
            if docs:
                doc_ref = docs[0].reference
            else:
                # create the doc if the doc does not exist
                doc_ref = self.checkpoint_writes_collection.document()
            doc = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "idx": idx,
                "channel": channel,
                "type": type_,
                "value": serialized_value,
            }
            self.batch.set(doc_ref, doc, merge=True)
        await self.batch.commit()

# endregion: checkpoint saver

# region: compile graph
graph_builder = StateGraph(State)

graph_builder.add_node("identify_image_items", identify_image_items)
graph_builder.add_node(
    "rephrase_question_based_on_image_items", rephrase_question_based_on_image_items
)
graph_builder.add_node(
    "rephrase_question_based_on_chat_history", rephrase_question_based_on_chat_history
)
graph_builder.add_node("retrieve_docs", retrieve_docs)
graph_builder.add_node("grade_retrieved_docs", grade_retrieved_docs)
graph_builder.add_node("perform_web_search", perform_web_search)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("generate_answer_from_llm", generate_answer_from_llm)
graph_builder.add_node("persist_chat_messages", persist_chat_messages)

graph_builder.add_conditional_edges(
    START,
    is_image_present,
    {
        "has_image": "identify_image_items",
        "no_image": "rephrase_question_based_on_chat_history",
    },
)
graph_builder.add_edge("identify_image_items", "rephrase_question_based_on_image_items")
graph_builder.add_edge(
    "rephrase_question_based_on_image_items", "rephrase_question_based_on_chat_history"
)
graph_builder.add_edge("rephrase_question_based_on_chat_history", "retrieve_docs")
graph_builder.add_edge("retrieve_docs", "grade_retrieved_docs")

graph_builder.add_edge("perform_web_search", "grade_retrieved_docs")

graph_builder.add_conditional_edges(
    "grade_retrieved_docs",
    should_do_web_search,
    {
        "web_search_not_needed": "generate_answer",
        "web_search_needed": "perform_web_search",
        "stop_web_search": "generate_answer_from_llm",
    },
)

graph_builder.add_edge("generate_answer", "persist_chat_messages")
graph_builder.add_edge("generate_answer_from_llm", "persist_chat_messages")
graph_builder.add_edge("persist_chat_messages", END)

    
# endregion: compile graph
