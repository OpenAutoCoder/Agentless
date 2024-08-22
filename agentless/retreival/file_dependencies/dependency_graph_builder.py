import os
from typing import Dict, List, Union
from tree_sitter import Node
from agentless.retreival.TextChunker.CodeParser import CodeParser

class DependencyGraphBuilder:

    def __init__(self, codebase: Dict[str, str]):
        self.codebase = codebase
        file_extensions = list(set([filename.split('.')[-1] for filename in codebase.keys()]))
        self.code_parser = CodeParser(file_extensions)
        self.dependency_graph = {}

    def build_complete_dependency_graph(self):
        """
        Builds both the forward and reverse dependency graphs for the codebase.
        """
        self.build_dependency_graph()
        self.build_reverse_dependencies()

    def build_dependency_graph(self) -> None:
        """Builds a list of the files that each file depends on."""
        for filename, content in self.codebase.items():
            try:
                file_extension = filename.split('.')[-1]
                if file_extension not in self.code_parser.language_extension_map:
                    print(f"Skipping unsupported file type: {filename}")
                    continue

                tree = self._parse_code(content, file_extension)
                if tree is None:
                    print(f"Skipping file due to parsing error: {filename}")
                    continue

                self.dependency_graph[filename] = {
                    'depends_on': {
                        'files': [],
                        'external': [],
                        'imported_entities': {}  # Renamed to 'imported_entities'
                    },
                    'depended_by': []
                }
                self._extract_depends_on(tree, filename)

            except Exception as e:
                print(f"An error occurred while processing {filename}: {str(e)}")
                continue

    def _extract_depends_on(self, node: Node, file_path: str) -> None:
        file_dir = os.path.dirname(file_path)
        file_extension = os.path.splitext(file_path)[-1]

        if file_extension == '.py':
            if node.type in ['import_statement', 'import_from_statement', 'import']:
                if node.type == 'import_from_statement':
                    # Handle 'from ... import ...' syntax
                    from_module = node.children[1].text.decode("utf-8")
                    absolute_import_path = self.resolve_path(file_dir, from_module)
                    file_dependency = self.find_file_from_import_path(absolute_import_path)
                    
                    # Extract imported entities
                    imported_entities = []
                    for child in node.children[3:]:
                        if child.type in ['dotted_name', 'name']:
                            imported_entities.append(child.text.decode("utf-8"))
                                                   
                    if file_dependency and file_dependency in self.codebase:
                        self.dependency_graph[file_path]['depends_on']['files'].append(file_dependency)
                        self.dependency_graph[file_path]['depends_on']['imported_entities'][file_dependency] = imported_entities
                    else:
                        self.dependency_graph[file_path]['depends_on']['external'].append(from_module)
                        self.dependency_graph[file_path]['depends_on']['imported_entities'][from_module] = imported_entities
                else:
                    # Handle 'import ...' syntax
                    for child in node.children:
                        if child.type in ['dotted_name', 'name']:
                            import_path = child.text.decode("utf-8")
                            absolute_import_path = self.resolve_path(file_dir, import_path)
                            file_dependency = self.find_file_from_import_path(absolute_import_path)
                            if file_dependency and file_dependency in self.codebase:
                                self.dependency_graph[file_path]['depends_on']['files'].append(file_dependency)
                            else:
                                self.dependency_graph[file_path]['depends_on']['external'].append(import_path)

        elif file_extension == '.js':
            if node.type == 'import_statement':
                for child in node.children:
                    if child.type == 'string':
                        import_path = child.text.decode("utf-8").strip('\'\"')
                        absolute_import_path = self.make_absolute_path(file_dir, import_path)
                        file_dependency = self.find_file_from_import_path(absolute_import_path)
                        if file_dependency and file_dependency in self.codebase:
                            self.dependency_graph[file_path]['depends_on']['files'].append(file_dependency)
                        else:
                            self.dependency_graph[file_path]['depends_on']['external'].append(import_path)

        for child in node.children:
            self._extract_depends_on(child, file_path)

    def resolve_path(self, file_dir: str, import_path: str) -> str:
        # Convert file paths to a standard format
        standardized_path = os.path.normpath(import_path).replace("\\\\", "/")
        standardized_path = standardized_path.replace("\\", "/")
        standardized_path = standardized_path.replace(".", "/")

        # print(f"Standardized path: {standardized_path}")

        if standardized_path in self.codebase.keys():
            # If the import path is a file in the codebase, return the absolute path
            absolute_path = standardized_path
        else:
            # Resolve relative path to absolute path
            absolute_path = os.path.normpath(os.path.join(file_dir, standardized_path))

        return absolute_path

    def find_file_from_import_path(self, absolute_import_path: str):
        if absolute_import_path in self.codebase:
            return absolute_import_path
        
        else:    
            # strip the file extensions from the dictionary keys
            file_name = absolute_import_path.split('/')[-1].split('.')[0]
            for file_path, content in self.codebase.items():
                if file_path.split('/')[-1].split('.')[0] == file_name:
                    return file_path
            return None

    def make_absolute_path(self, file_dir: str, relative_path: str) -> str:
        try:
            path = os.path.normpath(os.path.join(file_dir, relative_path))
            return path.replace("\\", "/")
        except Exception as e:
            # print(f"Error making absolute path: {e}")
            pass

    
    def build_reverse_dependencies(self) -> None:
        """Builds a list of the files that each file is depended on by."""
        # Create a list of items to iterate over to prevent RuntimeError
        items = list(self.dependency_graph.items())
        for file, info in items:
            for dependency in info['depends_on']['files']:
                if dependency in self.dependency_graph:
                    self.dependency_graph[dependency]['depended_by'].append(file)
                else:
                    # Handle the case where the dependency is an external module
                    if dependency not in self.dependency_graph:
                        self.dependency_graph[dependency] = {'depends_on': {'files': [], 'external': [], 'imported_entities': {}}, 'depended_by': [file]}

    def find_dependents_of_file(self, filename: str) -> List[str]:
        return self.dependency_graph.get(filename, {}).get('depended_by', [])

    def find_dependencies_of_file(self, filename: str) -> Dict[str, List[str]]:
        return self.dependency_graph.get(filename, {}).get('depends_on', {'files': [], 'external': [], 'imported_entities': {}})

    def _parse_code(self, code: str, file_extension: str) -> Union[None, Node]:
        try:
            return self.code_parser.parse_code(code, file_extension)
        except Exception as e:
            print(f"An error occurred while parsing code: {str(e)}")
            return None

    def get_relevant_code_files_from_dependency_graph(self, coding_file_name: str) -> List[str]:
        """
        This function creates a list of filenames that are related to the coding file.
        It starts by populating the list with files this file depends on and then
        adds files that depend on those files.
        :return: List of filenames that are related to the coding file.
        """
        # Get the files that the coding file depends on
        relevant_files = set(self.find_dependencies_of_file(coding_file_name)['files'])

        # Get the files that depend on the files that the coding file depends on
        for file in list(relevant_files):  # Iterate over a copy of the set to avoid modifying it during iteration
            relevant_files.update(self.find_dependents_of_file(file))

        # Remove the coding file from the set, if it's present
        relevant_files.discard(coding_file_name)  # Using discard to avoid KeyError if the item is not present

        return list(relevant_files)