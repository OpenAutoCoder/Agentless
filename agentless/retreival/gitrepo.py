import os
from pathlib import Path
from typing import Iterable, Dict
import git
from agentless.retreival import config


class ObjectNotFound(RuntimeError):
    """Exception raised when a requested object is not found in the repository."""
    pass

class FileStorage():
    """
    Abstract base class representing a file storage system.
    This class defines the interface for a file storage system, which includes
    methods for retrieving files and their contents.
    """

    def get_files(self) -> Iterable[str]:
        """
        Abstract method to retrieve all files in the storage system.
        This method should return an iterable of SrcFile objects representing
        all the files in the storage system.
        
        Returns:
            Iterable[SrcFile]: An iterable of SrcFile objects.
        """
        raise NotImplementedError()

    def get_file_content(self, path: str) -> str:
        """
        Abstract method to retrieve the content of a file by its unique identifier.
        This method should return the content of the file as a string.
        
        Args:
            file_id (str): The unique identifier of the file.
        
        Returns:
            str: The content of the file.
        """
        raise NotImplementedError()


    def get_file_sha(self, path: str) -> str:
        raise NotImplementedError()

    def save_file(self, path: str, content: str):
        raise NotImplementedError()

    def get_diff_patch(self) -> str:
        raise NotImplementedError()


def get_files_as_dict(fs: FileStorage) -> Dict[str, str]:
    return {path: fs.get_file_content(path) for path in fs.get_files()}



class GitRepo(FileStorage):
    """
    A class to represent a Git repository and provide access to its files.

    This class implements the FileStorage interface and provides methods to
    retrieve files and their contents from a Git repository.

    Attributes:
        tree (git.Tree): The tree object representing the state of the repository at a specific commit or branch.
    """

    def __init__(self, name: str):
        """
        Initialize the GitRepo object.

        Args:
            repo_dir (Path): The directory where the Git repository is located.
            branch_or_commit (str): The branch name or commit hash to initialize the repository to.

        Raises:
            git.exc.InvalidGitRepositoryError: If the provided directory is not a valid Git repository.
            git.exc.BadName: If the provided branch or commit does not exist in the repository.
        """
        self.name = name
        self.repo_dir = config.GITHUB_REPOS_DIR / name
        self._file_shas = None

    @property
    def repo(self):
        return git.Repo(self.repo_dir)

    @property
    def file_shas(self):
        if self._file_shas is None:
            self._file_shas = {blob.path: blob.hexsha for _, blob in self.repo.index.iter_blobs()}
        return self._file_shas


    def get_files(self) -> Iterable[str]:
        """
        Retrieve all files in the repository.

        This method traverses the repository tree and yields GitSrcFile objects
        representing each file in the repository.

        Returns:
            Iterable[SrcFile]: An iterable of GitSrcFile objects representing the files in the repository.
        """
        for path in self.file_shas:
            if not ('node_modules' in path or 'venv' in path):
                _, file_extension = os.path.splitext(path)
                if file_extension in ['.js', '.py', '.css', '.html']:
                    yield path

    def get_file_content(self, path: str) -> str:
        """
        Retrieve the content of a file by its unique identifier.

        Args:
            file_id (str): The unique identifier (SHA-1 hash) of the file.

        Returns:
            str: The content of the file as a UTF-8 decoded string.

        Raises:
            ObjectNotFound: If the file with the specified identifier is not found in the repository.
        """
        full_path = self.repo_dir / path
        if full_path.exists():
            try:
                # Try UTF-8 first
                return full_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try with 'latin-1' encoding
                return full_path.read_text(encoding='latin-1')
        raise ObjectNotFound(f"Cannot find object with path: {path}")

    def get_file_sha(self, path: str) -> str:
        try:
            return self.file_shas[path]
        except KeyError:
            raise ObjectNotFound(f"Cannot find object with path: {path}")

    def save_file(self, path: str, content: str):
        full_path = self.repo_dir / path
        full_path.write_text(content)
        self.repo.index.add([path])
        entry = self.repo.index.entries[(path, 0)]
        self.file_shas[entry.path] = entry.hexsha

    def get_code_dict(self):
        return {path: self.get_file_content(path) for path in self.get_files()}

    def get_diff_patch(self):
        return self.repo.git.diff('HEAD')
    
    def get_file_names_changed_between_commits(self, commit1: str, commit2: str):
        """
        Get the names of files that have changed between two commits.

        Args:
            commit1 (str): The SHA or reference of the first commit.
            commit2 (str): The SHA or reference of the second commit.

        Returns:
            str: A string containing the names of the files that have changed,
                 with each filename on a new line.

        Note:
            This method uses the 'git diff' command with the '--name-only' option
            to list only the names of the files that have changed between the
            two specified commits.
        """
        return self.repo.git.diff(commit1, commit2, name_only=True)