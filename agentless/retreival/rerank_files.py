import cohere
import os
from typing import Dict, List

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
# init client
co = cohere.Client(os.environ["COHERE_API_KEY"])


def rerank_files(query: str, code_files: Dict[str, str]) -> Dict[str, str]:
    """
    Reranks a list of code files based on their relevance to a given query.

    This function uses the Cohere API to rerank the provided code files based on their
    relevance to the input query. It returns a dictionary of filenames and contents sorted by relevance.

    Args:
        query (str): The query string used to rank the code files.
        code_files (Dict[str, str]): A dictionary where keys are filenames and values are file contents.

    Returns:
        Dict[str, str]: A dictionary of filenames and contents sorted by their relevance to the query.

    Note:
        This function requires a valid Cohere API key to be set in the environment variables.
    """
    # Check if code_files is empty
    if not code_files:
        print("Warning: No code files provided for reranking. Returning empty dictionary.")
        return {}
   
    # Prepare the documents for reranking
    documents = list(code_files.values())
    filenames = list(code_files.keys())

    # Rerank the documents
    reranked_results = co.rerank(
        query=query,
        documents=documents,
        model="rerank-english-v2.0"
    )

    # Create a reranked dictionary of filenames and contents
    reranked_files = {filenames[r.index]: documents[r.index] for r in reranked_results.results}

    return reranked_files


def test_rerank_files():
    # Mock query and code files
    query = "Find the maximum value in an array"
    code_files = {
        "file1.py": "def sort_list(lst): return sorted(lst)",
        "file2.py": "def find_max(arr): return max(arr)",
        "file3.py": "def calculate_average(numbers): return sum(numbers) / len(numbers)"
    }

    # Call the rerank_files function
    reranked_files = rerank_files(query, code_files)
    print(reranked_files)

    # Assertions
    assert isinstance(reranked_files, dict), "rerank_files should return a dictionary"
    assert len(reranked_files) == len(code_files), "reranked dict should have the same length as input"
    assert set(reranked_files.keys()) == set(code_files.keys()), "reranked dict should contain all original filenames"
    assert set(reranked_files.values()) == set(code_files.values()), "reranked dict should contain all original file contents"
    
    # Check if the most relevant file is ranked first
    assert list(reranked_files.keys())[0] == "file2.py", "Expected 'file2.py' to be ranked first"

    print("All tests passed!")

# Run the test
if __name__ == "__main__":
    test_rerank_files()