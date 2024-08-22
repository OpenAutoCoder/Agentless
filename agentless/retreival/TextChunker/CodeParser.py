import os
import subprocess
from typing import List, Dict, Union, Tuple
from tree_sitter import Language, Parser, Node
import logging

class CodeParser:
    # Added a CACHE_DIR class attribute for caching
    CACHE_DIR = os.path.expanduser("~/.code_parser_cache")

    def __init__(self, file_extensions: Union[None, List[str], str] = None):
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]
        self.language_extension_map = {
            "py": "python",
            "js": "javascript",
            "jsx": "javascript",
            "css": "css",
            "ts": "typescript",
            "tsx": "typescript",
            "php": "php",
            "rb": "ruby"
        }
        if file_extensions is None:
            self.language_names = []
        else:
            self.language_names = [self.language_extension_map.get(ext) for ext in file_extensions if
                                   ext in self.language_extension_map]
        self.languages = {}
        self._install_parsers()

    def _install_parsers(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        try:
            # Ensure cache directory exists
            if not os.path.exists(self.CACHE_DIR):
                os.makedirs(self.CACHE_DIR)

            for language in self.language_names:
                repo_path = os.path.join(self.CACHE_DIR, f"tree-sitter-{language}")

                # Check if the repository exists and contains necessary files
                if not os.path.exists(repo_path) or not self._is_repo_valid(repo_path, language):
                    try:
                        if os.path.exists(repo_path):
                            logging.info(f"Updating existing repository for {language}")
                            update_command = f"cd {repo_path} && git pull"
                            subprocess.run(update_command, shell=True, check=True)
                        else:
                            logging.info(f"Cloning repository for {language}")
                            clone_command = f"git clone https://github.com/tree-sitter/tree-sitter-{language} {repo_path}"
                            subprocess.run(clone_command, shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Failed to clone/update repository for {language}. Error: {e}")
                        continue

                try:
                    build_path = os.path.join(self.CACHE_DIR, f"build/{language}.so")
                    
                    # Special handling for TypeScript
                    if language == 'typescript':
                        ts_dir = os.path.join(repo_path, 'typescript')
                        tsx_dir = os.path.join(repo_path, 'tsx')
                        if os.path.exists(ts_dir) and os.path.exists(tsx_dir):
                            Language.build_library(build_path, [ts_dir, tsx_dir])
                        else:
                            raise FileNotFoundError(f"TypeScript or TSX directory not found in {repo_path}")
                    if language == 'php':
                        php_dir = os.path.join(repo_path, 'php')
                        Language.build_library(build_path, [php_dir])
                    else:
                        Language.build_library(build_path, [repo_path])
                    
                    self.languages[language] = Language(build_path, language)
                    logging.info(f"Successfully built and loaded {language} parser")
                except Exception as e:
                    logging.error(f"Failed to build or load language {language}. Error: {str(e)}")

        except Exception as e:
            logging.error(f"An unexpected error occurred during parser installation: {str(e)}")

    def _is_repo_valid(self, repo_path: str, language: str) -> bool:
        """Check if the repository contains necessary files."""
        if language == 'typescript':
            return (os.path.exists(os.path.join(repo_path, 'typescript', 'src', 'parser.c')) and
                     os.path.exists(os.path.join(repo_path, 'tsx', 'src', 'parser.c')))
        elif language == 'php':
            return os.path.exists(os.path.join(repo_path, 'php', 'src', 'parser.c'))
        else:
            return os.path.exists(os.path.join(repo_path, 'src', 'parser.c'))

    def parse_code(self, code: str, file_extension: str) -> Union[None, Node]:
        language_name = self.language_extension_map.get(file_extension)
        if language_name is None:
            print(f"Unsupported file type: {file_extension}")
            return None

        language = self.languages.get(language_name)
        if language is None:
            print("Language parser not found")
            return None

        parser = Parser()
        parser.set_language(language)
        tree = parser.parse(bytes(code, "utf8"))

        if tree is None:
            print("Failed to parse the code")
            return None

        return tree.root_node

    def extract_points_of_interest(self, node: Node, file_extension: str) -> List[Tuple[Node, str]]:
        node_types_of_interest = self._get_node_types_of_interest(file_extension)

        points_of_interest = []
        if node.type in node_types_of_interest.keys():
            points_of_interest.append((node, node_types_of_interest[node.type]))

        for child in node.children:
            points_of_interest.extend(self.extract_points_of_interest(child, file_extension))

        return points_of_interest

    def _get_node_types_of_interest(self, file_extension: str) -> Dict[str, str]:
        node_types = {
            'py': {
                'import_statement': 'Import',
                'export_statement': 'Export',
                'class_definition': 'Class',
                'function_definition': 'Function',
            },
            'css': {
                'tag_name': 'Tag',
                '@media': 'Media Query',
            },
            'js': {
                'import_statement': 'Import',
                'export_statement': 'Export',
                'class_declaration': 'Class',
                'function_declaration': 'Function',
                'arrow_function': 'Arrow Function',
                'statement_block': 'Block',
            },
            'ts': {
                'import_statement': 'Import',
                'export_statement': 'Export',
                'class_declaration': 'Class',
                'function_declaration': 'Function',
                'arrow_function': 'Arrow Function',
                'statement_block': 'Block',
                'interface_declaration': 'Interface',
                'type_alias_declaration': 'Type Alias',
            },
            'php': {
                'namespace_definition': 'Namespace',
                'class_declaration': 'Class',
                'method_declaration': 'Method',
                'function_definition': 'Function',
                'interface_declaration': 'Interface',
                'trait_declaration': 'Trait',
            },
            'rb': {
                'class': 'Class',
                'method': 'Method',
                'module': 'Module',
                'singleton_class': 'Singleton Class',
                'begin': 'Begin Block',
            }
        }

        if file_extension in node_types.keys():
            return node_types[file_extension]
        elif file_extension == "jsx":
            return node_types["js"]
        elif file_extension == "tsx":
            return node_types["ts"]
        else:
            raise ValueError("Unsupported file type")
        

    def _get_nodes_for_comments(self, file_extension: str) -> Dict[str, str]:
        node_types = {
            'py': {
                'comment': 'Comment',
                'decorator': 'Decorator',  # Broadened category
            },
            'css': {
                'comment': 'Comment'
            },
            'js': {
                'comment': 'Comment',
                'decorator': 'Decorator',  # Broadened category
            },
            'ts': {
                'comment': 'Comment',
                'decorator': 'Decorator',
            },
            'php': {
                'comment': 'Comment',
                'attribute': 'Attribute',
            },
            'rb': {
                'comment': 'Comment',
            }
        }

        if file_extension in node_types.keys():
            return node_types[file_extension]
        elif file_extension == "jsx":
            return node_types["js"]
        else:
            raise ValueError("Unsupported file type")
        
    def extract_comments(self, node: Node, file_extension: str) -> List[Tuple[Node, str]]:
        node_types_of_interest = self._get_nodes_for_comments(file_extension)

        comments = []
        if node.type in node_types_of_interest:
            comments.append((node, node_types_of_interest[node.type]))

        for child in node.children:
            comments.extend(self.extract_comments(child, file_extension))

        return comments

    def get_lines_for_points_of_interest(self, code: str, file_extension: str) -> List[int]:
        language_name = self.language_extension_map.get(file_extension)
        if language_name is None:
            raise ValueError("Unsupported file type")

        language = self.languages.get(language_name)
        if language is None:
            raise ValueError("Language parser not found")

        parser = Parser()
        parser.set_language(language)

        tree = parser.parse(bytes(code, "utf8"))

        root_node = tree.root_node
        points_of_interest = self.extract_points_of_interest(root_node, file_extension)

        line_numbers_with_type_of_interest = {}

        for node, type_of_interest in points_of_interest:
            start_line = node.start_point[0] 
            if type_of_interest not in line_numbers_with_type_of_interest:
                line_numbers_with_type_of_interest[type_of_interest] = []

            if start_line not in line_numbers_with_type_of_interest[type_of_interest]:
                line_numbers_with_type_of_interest[type_of_interest].append(start_line)

        lines_of_interest = []
        for _, line_numbers in line_numbers_with_type_of_interest.items():
            lines_of_interest.extend(line_numbers)

        return lines_of_interest

    def get_lines_for_comments(self, code: str, file_extension: str) -> List[int]:
        language_name = self.language_extension_map.get(file_extension)
        if language_name is None:
            raise ValueError("Unsupported file type")

        language = self.languages.get(language_name)
        if language is None:
            raise ValueError("Language parser not found")

        parser = Parser()
        parser.set_language(language)

        tree = parser.parse(bytes(code, "utf8"))

        root_node = tree.root_node
        comments = self.extract_comments(root_node, file_extension)

        line_numbers_with_comments = {}

        for node, type_of_interest in comments:
            start_line = node.start_point[0] 
            if type_of_interest not in line_numbers_with_comments:
                line_numbers_with_comments[type_of_interest] = []

            if start_line not in line_numbers_with_comments[type_of_interest]:
                line_numbers_with_comments[type_of_interest].append(start_line)

        lines_of_interest = []
        for _, line_numbers in line_numbers_with_comments.items():
            lines_of_interest.extend(line_numbers)

        return lines_of_interest

    def print_all_line_types(self, code: str, file_extension: str):
        language_name = self.language_extension_map.get(file_extension)
        if language_name is None:
            print(f"Unsupported file type: {file_extension}")
            return

        language = self.languages.get(language_name)
        if language is None:
            print("Language parser not found")
            return

        parser = Parser()
        parser.set_language(language)
        tree = parser.parse(bytes(code, "utf8"))

        root_node = tree.root_node
        line_to_node_type = self.map_line_to_node_type(root_node)

        code_lines = code.split('\n')

        for line_num, node_types in line_to_node_type.items():
            line_content = code_lines[line_num - 1]  # Adjusting index for zero-based indexing
            print(f"line {line_num}: {', '.join(node_types)} | Code: {line_content}")


    def map_line_to_node_type(self, node, line_to_node_type=None, depth=0):
        if line_to_node_type is None:
            line_to_node_type = {}

        start_line = node.start_point[0] + 1  # Tree-sitter lines are 0-indexed; converting to 1-indexed

        # Only add the node type if it's the start line of the node
        if start_line not in line_to_node_type:
            line_to_node_type[start_line] = []
        line_to_node_type[start_line].append(node.type)

        for child in node.children:
            self.map_line_to_node_type(child, line_to_node_type, depth + 1)

        return line_to_node_type
    