import hashlib
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai
import uuid
from collections import defaultdict
import json
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
import hashlib
import os
from agentless.retreival.TextChunker.count_tokens import count_tokens
from agentless.retreival.TextChunker.Chunker import CodeChunker
from agentless.retreival.TextChunker.Chunker import ChunkDictionary


EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
if not EMBEDDINGS_DIR.exists():
    EMBEDDINGS_DIR.mkdir(parents=True)


COST_PER_TOKEN = 0.0000001

load_dotenv()


def get_file_shas(code_files: Dict[str, str]) -> Dict[str, str]:
    """
    Generate SHA-256 hashes for the contents of the given code files.
    
    Args:
        code_files (Dict[str, str]): A dictionary of file paths and their contents.
    
    Returns:
        Dict[str, str]: A dictionary mapping file paths to their SHA-256 hashes.
    """
    return {file_path: hashlib.sha256(content.encode()).hexdigest() 
            for file_path, content in code_files.items()}
    
class VectorDB:
    """
    Abstract base class for vector database operations.

    This class defines the interface for vector database operations that should be
    implemented by specific vector database implementations.
    """

    def create_collection_if_not_exists(self):
        """
        Creates a new collection in the vector database if it doesn't already exist.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def rename_collection(self, new_name: str):
        """
        Renames the current collection to the specified new name.

        Args:
            new_name (str): The new name for the collection.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def upload_data_to_db(self, code_files: Dict[str, str]):
        """
        Uploads code data to the vector database.

        Args:
            code_files (Dict[str, str]): A dictionary of file paths and their contents.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def search_code(self, query: str, top_n: int = 15, files_to_exclude_from_search: List[str] = None) -> List[Dict[str, Any]]:
        """
        Searches the code in the vector database.

        Args:
            query (str): The search query.
            top_n (int): The number of top results to return.
            files_to_exclude_from_search (List[str]): A list of file paths to exclude from the search.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def delete_all_points(self, force: bool = False):
        """
        Deletes all points in the collection.

        Args:
            force (bool): If True, skip the confirmation prompt.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _filter_code_files(self, code_files: Dict[str, str]) -> Dict[str, str]:
        """
        Filters out code files that already exist in the vector database.

        Args:
            code_files (Dict[str, str]): A dictionary of file paths and their contents.

        Returns:
            Dict[str, str]: A dictionary of file paths and contents for files that don't exist in the database.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")



class QdrantDB(VectorDB):
    """
    VectorOperations class handles vector database operations for code repositories.

    This class provides functionality to:
    - Create and manage collections in a Qdrant vector database
    - Upload code data (file contents, embeddings) to the database
    - Search for relevant code snippets based on queries
    - Manage and update the vector database as the code repository changes

    It uses OpenAI's embedding model to generate vector representations of code snippets
    and Qdrant for efficient similarity search operations.
    """
    def __init__(self, repo_name: str, codebase_dict: Dict[str, str], include_chunk_content: bool = True, regenerate_embeddings: bool = False):
        self.client = QdrantClient(
            host=os.getenv("QDRANT_HOST"),
            port=6333,
            api_key=os.getenv("QDRANT_API_KEY"),
            https=True
        )
        self.collection_name = f"repo_{repo_name}"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.vector_size = 1536
        self.include_chunk_content = include_chunk_content
        self.codebase_dict = codebase_dict
        self.regenerate_embeddings = regenerate_embeddings

        # Create the collection if it doesn't exist
        self.create_collection_if_not_exists()
        if self.regenerate_embeddings:
            self.delete_all_points(force=True)
        self.upload_data_to_db(self.codebase_dict)

    def create_collection_if_not_exists(self):
        """
        Creates a new collection in the vector database if it doesn't already exist.

        This method checks if a collection with the name specified by self.collection_name
        exists in the Qdrant database. If it doesn't exist, it creates a new collection
        with the following configuration:
        - Name: self.collection_name
        - Vector size: self.vector_size (1536)
        - Distance metric: Cosine similarity

        The method uses the Qdrant client to interact with the database and create
        the collection if necessary.

        Returns:
            None
        """
        collections = self.client.get_collections().collections
        if not any(collection.name == self.collection_name for collection in collections):
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
            )

    def rename_collection(self, new_repo_name: str):
        new_collection_name = f"repo_{new_repo_name}"
        self.client.rename_collection(
            collection_name=self.collection_name,
            new_collection_name=new_collection_name
        )
        self.collection_name = new_collection_name
 
    def upload_data_to_db(self, code_files: Dict[str,str]):
        """
        Upload data to Qdrant
        """
        # get the file shas from the data. CONFIRM THIS IS CORRECT
        filtered_code_files = self._filter_code_files(code_files)

        # prepare the data for upload
        prepared_data = self._prepare_data_for_upload(filtered_code_files)

        # embed the data
        data_with_embeddings = self._bulk_embed(prepared_data)
    
        # upload the data to the vector database
        self._bulk_upload_data(data_with_embeddings)
        
        return {"message": "Data uploaded to Qdrant"}
    
    def _get_existing_points(self) -> List[Dict[str, str]]:
        """
        Get all the file shas and file paths that already exist in the vector database.
        """
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=[0] * self.vector_size,
                limit=100000
            )
            return [{'file_sha': hit.payload['file_sha'], 'file_path': hit.payload['file_path']} for hit in search_result]
        except Exception as e:
            print(f"Error in _get_existing_shas: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Collection name: {self.collection_name}")
            print(f"Vector size: {self.vector_size}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            return []
    
    def _filter_code_files(self, code_files: Dict[str, str]) -> Dict[str, str]:
        """
        Filter out code files that already exist in the vector database.

        Args:
            code_files (Dict[str, str]): A dictionary of file paths and their contents.

        Returns:
            Dict[str, str]: A dictionary of file paths and contents for files that don't exist in the database.
        """
        # Get a dictionary of file shas
        file_shas = get_file_shas(code_files)

        # Get all existing points from the database
        existing_points = self._get_existing_points()

        # Create a set of tuples (file_path, sha) for existing points
        existing_file_path_sha_pairs = {(point['file_path'], point['file_sha']) for point in existing_points}

        # Create a set of tuples (file_path, sha) for new files
        new_file_path_sha_pairs = {(file_path, file_shas[file_path]) for file_path in code_files}

        # Find the pairs that don't exist in the database
        pairs_to_add = new_file_path_sha_pairs - existing_file_path_sha_pairs

        # Filter the code_files dictionary
        filtered_files = {
            file_path: content
            for file_path, content in code_files.items()
            if (file_path, file_shas[file_path]) in pairs_to_add
        }

        filtered_out = set(code_files.keys()) - set(filtered_files.keys())

        print(f"Filtered out files: {len(filtered_out)} out of {len(code_files)} because they already exist in the database.")

        return filtered_files
    
    def _embed_content(self, content: str) -> List[float]:
        try:
            response = openai.embeddings.create(
                input=[content],
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error embedding chunk: {e}")
            return None
    
    def _bulk_embed(self, chunked_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Bulk embed a list of texts using OpenAI's text-embedding-ada-002 model with batching.
        Use cached embeddings when available and only generate new ones as needed.
        """
        MAX_TOKENS_PER_CHUNK = 8000
        cache_dir = Path(EMBEDDINGS_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{self.collection_name}_embeddings.pkl"
        
        # Load existing embeddings if file exists
        if cache_path.exists():
            with cache_path.open('rb') as f:
                saved_embeddings = pickle.load(f)
        else:
            saved_embeddings = {}

        new_chunks = []
        for chunk in chunked_data:
            key = chunk['file_path'] + str(chunk['chunk_number'])
            if key in saved_embeddings:
                chunk['embedding'] = saved_embeddings[key]
            else:
                new_chunks.append(chunk)

        print(f"Found {len(chunked_data) - len(new_chunks)} cached embeddings. Generating {len(new_chunks)} new embeddings.")

        if new_chunks:
            batches = []
            current_batch = []
            current_batch_tokens = 0

            for chunk in new_chunks:
                chunk_tokens = count_tokens(chunk['chunk_content'], "text-embedding-ada-002")
                if chunk_tokens > MAX_TOKENS_PER_CHUNK:
                    sub_chunks = self._split_large_chunk(chunk, MAX_TOKENS_PER_CHUNK)
                    batches.extend([[sub_chunk] for sub_chunk in sub_chunks])
                elif current_batch_tokens + chunk_tokens > MAX_TOKENS_PER_CHUNK:
                    batches.append(current_batch)
                    current_batch = [chunk]
                    current_batch_tokens = chunk_tokens
                else:
                    current_batch.append(chunk)
                    current_batch_tokens += chunk_tokens

            if current_batch:
                batches.append(current_batch)

            print(f"Embedding {len(new_chunks)} chunks in {len(batches)} batches...")

            for batch in tqdm(batches, desc="Embedding batches"):
                try:
                    response = openai.embeddings.create(
                        input=[chunk['chunk_content'] for chunk in batch],
                        model="text-embedding-3-large"
                    )
                    new_embeddings = [data.embedding for data in response.data]
                    
                    for chunk, embedding in zip(batch, new_embeddings):
                        key = chunk['file_path'] + str(chunk['chunk_number'])
                        saved_embeddings[key] = embedding
                        chunk['embedding'] = embedding
                
                except Exception as e:
                    print(f"Error during bulk embedding: {e}")
                    for chunk in batch:
                        chunk['embedding'] = None

            # Save updated embeddings to pickle file
            with cache_path.open('wb') as f:
                pickle.dump(saved_embeddings, f)

        return chunked_data

    def _split_large_chunk(self, chunk: Dict[str, Any], max_tokens: int) -> List[Dict[str, Any]]:
        """
        Split a large chunk into smaller sub-chunks that fit within the token limit.
        """
        content = chunk['chunk_content']
        sub_chunks = []
        while content:
            sub_content = content[:max_tokens]
            sub_chunk = chunk.copy()
            sub_chunk['chunk_content'] = sub_content
            sub_chunks.append(sub_chunk)
            content = content[max_tokens:]
        return sub_chunks

    def _prepare_data_for_upload(self, files: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Prepare data for upload by creating metadata and code chunks using multithreading.
        """
        files_by_extension = defaultdict(dict)

        # Sort files by extension
        for file_path, content in files.items():
            file_extension = os.path.splitext(file_path)[1].lstrip('.')
            files_by_extension[file_extension][file_path] = content

        print(f"Total files to process: {len(files)}")
        print(f"Files by extension: {', '.join(f'{ext}: {len(files)}' for ext, files in files_by_extension.items())}")

        # Build chunkers for all required extensions
        chunkers = {}
        for ext in files_by_extension.keys():
            try:
                chunkers[ext] = CodeChunker(ext)
            except ValueError:
                print(f"Warning: Parser for extension '{ext}' not available. Using ChunkDictionary instead.")
                chunkers[ext] = ChunkDictionary()

        prepared_data = []

        def process_file(file_path, content, chunker):
            file_sha = hashlib.sha256(content.encode()).hexdigest()
            try:
                chunks = chunker.chunk(content, token_limit=100)
                return [(file_path, file_sha, chunk_number, chunk_content) 
                        for chunk_number, chunk_content in chunks.items()]
            except Exception as e:
                print(f"Error chunking file {file_path}: {e}")
                return []

        with ThreadPoolExecutor() as executor:
            futures = []
            for ext, files in files_by_extension.items():
                chunker = chunkers[ext]
                for file_path, content in files.items():
                    futures.append(executor.submit(process_file, file_path, content, chunker))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                prepared_data.extend([{
                    'file_path': file_path,
                    'file_sha': file_sha,
                    'chunk_content': chunk_content,
                    'chunk_number': chunk_number
                } for file_path, file_sha, chunk_number, chunk_content in future.result()])

        return prepared_data

    def _bulk_upload_data(self, data_with_embeddings: List[Dict[str, Any]]):
        """
        Data to upload includes:
        - file_path
        - file_sha
        - chunk_number
        - embedding
        - chunk_content (optional)
        """

        def create_point(data_item):
            if 'embedding' not in data_item or data_item['embedding'] is None:
                print(f"Warning: Missing embedding for file {data_item['file_path']}, chunk {data_item['chunk_number']}")
                return None

            payload = {
                'file_path': data_item['file_path'],
                'file_sha': data_item['file_sha'],
                'chunk_number': data_item['chunk_number'],
            }
            if self.include_chunk_content:
                payload['chunk_content'] = data_item['chunk_content']

            return models.PointStruct(
                id=str(uuid.uuid4()),  # Generate a new UUID for each point
                vector=data_item['embedding'],
                payload=payload
            )
        
        # Create points with batching
        batch_size = 100  # Adjust this value based on your system's capabilities
        all_points = []
        
        print("Creating points...")
        for i in tqdm(range(0, len(data_with_embeddings), batch_size), desc="Creating points"):
            batch = data_with_embeddings[i:i+batch_size]
            batch_points = [create_point(data_item) for data_item in batch]
            all_points.extend([point for point in batch_points if point is not None])
        
        print(f"Created {len(all_points)} valid points")

        if not all_points:
            print("No valid points to upload.")
            return

        # Upload points in batches
        total_batches = len(all_points) // batch_size + (1 if len(all_points) % batch_size > 0 else 0)
        print(f"Uploading {total_batches} batches to Qdrant...")
        
        for i in tqdm(range(0, len(all_points), batch_size), desc="Uploading batches", total=total_batches):
            batch = all_points[i:i+batch_size]
            try:
                self.client.upload_points(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            except Exception as e:
                print(f"Error uploading batch: {e}")
        
        print("Upload complete")

    
    def search_code(self, query: str, top_n: int = 15, files_to_exclude_from_search: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search the code in the vector database and return a ranked list of unique file paths.
        """
        query_vector = self._embed_content(query)
        
        if files_to_exclude_from_search:
            filter_conditions = [models.FieldCondition(key="file_path", match=models.MatchValue(value=file_path)) for file_path in files_to_exclude_from_search]
        else:
            filter_conditions = []
        
        # Perform the search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=1000,  # Increase limit to get more results
            query_filter=models.Filter(
                must_not=filter_conditions
            )
        )   

        # Process results and keep the highest scoring chunk for each file
        file_scores = {}
        for hit in search_result:
            file_path = hit.payload.get('file_path')
            if file_path not in file_scores or hit.score > file_scores[file_path]['score']:
                file_scores[file_path] = {
                    'score': hit.score,
                    'chunk_number': hit.payload.get('chunk_number'),
                    'file_sha': hit.payload.get('file_sha'),
                }
                if self.include_chunk_content:
                    file_scores[file_path]['chunk_content'] = hit.payload.get('chunk_content')

        # Sort the results by score and return the top_n
        ranked_results = sorted(
            [{'file_path': fp, **data} for fp, data in file_scores.items()],
            key=lambda x: x['score'],
            reverse=True
        )[:top_n]

        return ranked_results


    def delete_all_points(self, force=False):
        """
        Delete all points in the collection.
        
        Args:
            force (bool): If True, skip the confirmation prompt.
        """
        if not force:
            confirmation = input(f"WARNING: You are about to delete all points in the collection '{self.collection_name}'.\n"
                                 f"This action is irreversible. Are you sure you want to proceed? (yes/no): ").lower()
            if confirmation != 'yes':
                print("Operation cancelled.")
                return

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                )
            )
            print(f"All points in collection '{self.collection_name}' have been deleted.")
        except Exception as e:
            print(f"Error deleting points from collection '{self.collection_name}': {str(e)}")


# Example usage
if __name__ == "__main__":
    # Load mock codefiles
    mock_codefiles_path = os.path.join(os.path.dirname(__file__), 'mock_codefiles.json')
    with open(mock_codefiles_path, 'r') as f:
        mock_codefiles = json.load(f)

    vector_ops = QdrantDB("test_collection", mock_codefiles, include_chunk_content=True, regenerate_embeddings=True)

    # Example: Perform a search
    query = "hello world"
    results_with_exclude = vector_ops.search_code(query, files_to_exclude_from_search=['simple.py'])
    print([result['file_path'] for result in results_with_exclude])
    print(f"Number of results with exclude: {len(results_with_exclude)}")
    assert len(results_with_exclude) == 15, f"Expected 15 results, but got {len(results_with_exclude)}"
    results_without_exclude = vector_ops.search_code(query)
    print([result['file_path'] for result in results_without_exclude])