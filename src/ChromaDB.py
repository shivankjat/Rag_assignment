import json
from typing import List

import chromadb
import pandas as pd
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
from Base import Logic
from utils import deterministic_uuid

default_ef = embedding_functions.DefaultEmbeddingFunction()

class ChromaDB_VectorStore(Logic):
    def __init__(self, config=None):
        Logic.__init__(self, config=config)
        if config is None:
            config = {}

        path = config.get("path", ".")
        self.embedding_function = config.get("embedding_function", default_ef)
        curr_client = config.get("client", "persistent")
        collection_metadata = config.get("collection_metadata", None)
        self.n_results_csv = config.get("n_results_csv", config.get("n_results", 10))
        
        if curr_client == "persistent":
            self.chroma_client = chromadb.PersistentClient(
                path=path, settings=Settings(anonymized_telemetry=False)
            )
        elif curr_client == "in-memory":
            self.chroma_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        elif isinstance(curr_client, chromadb.api.client.Client):
            self.chroma_client = curr_client
        else:
            raise ValueError(f"Unsupported client was set in config: {curr_client}")

        self.csv_collection = self.chroma_client.get_or_create_collection(
            name="csv",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = self.embedding_function([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding

    def add_csv(self, csv_file_path: str, **kwargs) -> str:
        metadata = self.get_csv_metadata_as_json(csv_file_path)
        id = deterministic_uuid(csv_file_path) + "-csv"
        self.csv_collection.add(
            documents=metadata,
            embeddings=self.generate_embedding(metadata),
            ids=id,
        )
        return id

    def get_csv_metadata_as_json(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        table_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        metadata = {
            "path": csv_file_path,
            "table_name": table_name,
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.apply(lambda x: str(x)).tolist(),
            "num_rows": len(df)
        }
        return json.dumps(metadata, indent=4)

    def get_related_csv(self, question: str, n_results: int = None, **kwargs) -> List[str]:
        if n_results is None:
            n_results = self.n_results_csv

        query_results = self.csv_collection.query(
            query_texts=[question],
            n_results=n_results,
        )

        related_documents = self._extract_answers(query_results)
        return related_documents

    @staticmethod
    def _extract_answers(query_results) -> List[str]:
        if query_results is None or "documents" not in query_results:
            return []

        documents = query_results["documents"]

        if len(documents) == 1 and isinstance(documents[0], list):
            try:
                documents = [json.loads(doc) for doc in documents[0]]
            except Exception:
                return documents[0]

        # Extract relevant fields to construct answers
        answers = []
        for doc in documents:
            doc_data = json.loads(doc)
            if "column_names" in doc_data and "table_name" in doc_data:
                answer = f"The table '{doc_data['table_name']}' contains columns: {', '.join(doc_data['column_names'])}."
                answers.append(answer)

        return answers
