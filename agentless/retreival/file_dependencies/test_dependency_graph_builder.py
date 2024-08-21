import unittest
from agentless.retreival.file_dependencies.dependency_graph_builder import DependencyGraphBuilder

class TestDependencyGraphBuilder(unittest.TestCase):

    def setUp(self):
        self.mock_codebase = {
            'main.py': 'import utils\nfrom models import User',
            'utils.py': 'from config import settings',
            'models.py': 'from database import db',
            'config.py': 'import os',
            'database.py': 'import sqlite3',
            'routes.py': 'from main import app \n import requests'
        }
        self.builder = DependencyGraphBuilder(self.mock_codebase)

    def test_build_complete_dependency_graph(self):
        self.builder.build_complete_dependency_graph()
        print(self.builder.dependency_graph)
        
        expected_graph = {
            'main.py': {'depends_on': {'files': ['utils.py', 'models.py'], 'external': []}, 'depended_by': ['routes.py']},
            'utils.py': {'depends_on': {'files': ['config.py'], 'external': []}, 'depended_by': ['main.py']},
            'models.py': {'depends_on': {'files': ['database.py'], 'external': []}, 'depended_by': ['main.py']},
            'config.py': {'depends_on': {'files': [], 'external': ['os']}, 'depended_by': ['utils.py']},
            'database.py': {'depends_on': {'files': [], 'external': ['sqlite3']}, 'depended_by': ['models.py']},
            'routes.py': {'depends_on': {'files': ['main.py'], 'external': ['requests']}, 'depended_by': []}
        }
        
        self.assertEqual(self.builder.dependency_graph, expected_graph)

    def test_find_dependents_of_file(self):
        self.builder.build_complete_dependency_graph()
        dependents = self.builder.find_dependents_of_file('utils.py')
        self.assertEqual(dependents, ['main.py'])

    def test_find_dependencies_of_file(self):
        self.builder.build_complete_dependency_graph()
        dependencies = self.builder.find_dependencies_of_file('main.py')
        expected_dependencies = {'files': ['utils.py', 'models.py'], 'external': []}
        self.assertEqual(dependencies, expected_dependencies)

    def test_get_relevant_code_files_from_dependency_graph(self):
        self.builder.build_complete_dependency_graph()
        print(self.builder.dependency_graph)
        relevant_files = self.builder.get_relevant_code_files_from_dependency_graph('main.py')
        expected_files = ['utils.py','models.py']
        self.assertCountEqual(relevant_files, expected_files)

    def test_get_external_dependencies_or_modules(self):
        self.builder.build_complete_dependency_graph()
        external_dependencies = set()
        for file_info in self.builder.dependency_graph.values():
            external_dependencies.update(file_info['depends_on']['external'])
        expected_external_dependencies = {'os', 'sqlite3', 'requests'}
        self.assertEqual(external_dependencies, expected_external_dependencies)

if __name__ == '__main__':
    unittest.main()