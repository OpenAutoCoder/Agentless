import unittest
from unittest.mock import patch
import tiktoken
import json
from agentless.retreival.TextChunker.Chunker import Chunker, CodeChunker

mock_codebase = json.load(open('agentless/retreival/mock_codefiles.json'))
                                
# Mocking the count_tokens function as it's external and not the focus of these tests
def mock_count_tokens(string: str, encoding_name='gpt-4') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Python Test Class
class TestCodeChunkerPython(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('agentless.retreival.TextChunker.count_tokens', side_effect=mock_count_tokens)
        self.mock_count_tokens = self.patcher.start()
        self.code_chunker = CodeChunker(file_extension='py')
        self.mock_codebase = mock_codebase
        
    def tearDown(self):
        self.patcher.stop()

    def test_chunk_simple_code(self):
        py_code = self.mock_codebase['simple.py']
        first_chunk_token_limit = mock_count_tokens("import sys")
        print(f"first_chunk_token_limit = {first_chunk_token_limit}")
        chunks = self.code_chunker.chunk(py_code, token_limit=25)
        token_count = self.mock_count_tokens(py_code)
        print(f"token_count = {token_count}")
        print(f"original code:\n {py_code}")
        Chunker.print_chunks(chunks)
        full_code = Chunker.consolidate_chunks_into_file(chunks)
        print(f"code after consolidation:\n {full_code}")
        num_lines = Chunker.count_lines(full_code)
        self.assertEqual(num_lines, len(py_code.split("\n"))) # The number of lines should be the same
        self.assertIn(full_code, py_code) # The full code should be in the original code
        self.assertEqual(len(chunks), 2) # There should be 2 chunks
        self.assertIn("import sys", chunks[1]) # The first chunk should contain the import statement
        self.assertIn("print('Hello, world!')", chunks[2]) # The second chunk should contain the print statement

    def test_chunk_code_text_only(self):
        py_code = self.mock_codebase['text_only.py']
        chunks = self.code_chunker.chunk(py_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(py_code.split("\n"))) # The number of lines should be the same
        self.assertIn(py_code, final_code) # The full code should be in the original code
        self.assertEqual(len(chunks), 1)
        self.assertIn("This file is empty and should test the chunker's ability to handle empty files", chunks[1])


    def test_chunk_code_with_routes(self):
        py_code = self.mock_codebase['routes.py']
        chunks = self.code_chunker.chunk(py_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(py_code.split("\n"))) # The number of lines should be the same
        self.assertIn(py_code, final_code) # The full code should be in the original code


    def test_chunk_code_with_models(self):
        py_code = self.mock_codebase['models.py']
        chunks = self.code_chunker.chunk(py_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(py_code.split("\n")))
        self.assertIn(py_code, final_code)

    def test_chunk_code_with_main(self):
        py_code = self.mock_codebase['main.py']
        chunks = self.code_chunker.chunk(py_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(py_code.split("\n")))
        self.assertIn(py_code, final_code)

    def test_chunk_code_with_utilities(self):
        py_code = self.mock_codebase['utilities.py']
        chunks = self.code_chunker.chunk(py_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(py_code.split("\n")))
        self.assertIn(py_code, final_code)

    def test_chunk_code_with_big_class(self):
        py_code = self.mock_codebase['big_class.py']
        chunks = self.code_chunker.chunk(py_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(py_code.split("\n")))
        self.assertIn(py_code, final_code)

# JavaScript Test Class
class TestCodeChunkerJavaScript(unittest.TestCase):

    def setUp(self):
        self.patcher = patch('agentless.retreival.TextChunker.count_tokens', side_effect=mock_count_tokens)
        self.mock_count_tokens = self.patcher.start()
        self.code_chunker = CodeChunker(file_extension='js')
        self.mock_codebase = mock_codebase

    def tearDown(self):
        self.patcher.stop()

    def test_chunk_javascript_simple_code(self):
        js_code = self.mock_codebase['simple.js']
        chunks = self.code_chunker.chunk(js_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(js_code.split("\n")))
        self.assertIn(js_code, final_code)


    def test_chunk_javascript_with_routes(self):
        js_code = self.mock_codebase['routes.js']
        chunks = self.code_chunker.chunk(js_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(js_code.split("\n")))
        self.assertIn(js_code, final_code)


    def test_chunk_javascript_with_models(self):
        js_code = self.mock_codebase['models.js']
        chunks = self.code_chunker.chunk(js_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(js_code.split("\n")))
        self.assertIn(js_code, final_code)

    def test_chunk_javascript_with_main(self):
        js_code = self.mock_codebase['main.js']
        chunks = self.code_chunker.chunk(js_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(js_code.split("\n")))
        self.assertIn(js_code, final_code)

    def test_chunk_javascript_with_utilities(self):
        js_code = self.mock_codebase['utilities.js']
        chunks = self.code_chunker.chunk(js_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(js_code.split("\n")))
        self.assertIn(js_code, final_code)

    def test_chunk_javascript_with_big_class(self):
        js_code = self.mock_codebase['big_class.js']
        chunks = self.code_chunker.chunk(js_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(js_code.split("\n")))
        self.assertIn(js_code, final_code)

    def test_chunk_javascript_with_react_component(self):
        js_code = self.mock_codebase['react_component.js']
        chunks = self.code_chunker.chunk(js_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(js_code.split("\n")))
        self.assertIn(js_code, final_code)

# CSS Test Class
class TestCodeChunkerCSS(unittest.TestCase):
   
    def setUp(self):
        self.patcher = patch('agentless.retreival.TextChunker.count_tokens', side_effect=mock_count_tokens)
        self.mock_count_tokens = self.patcher.start()
        self.code_chunker = CodeChunker(file_extension='css')
        self.mock_codebase = mock_codebase

    def tearDown(self):
        self.patcher.stop()

    def test_chunk_css_with_media_query(self):
        css_code = self.mock_codebase['media_queries.css']
        chunks = self.code_chunker.chunk(css_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(css_code.split("\n")))
        self.assertIn(css_code, final_code)

    def test_chunk_css_with_simple_css(self):
        css_code = self.mock_codebase['simple_styles.css']
        chunks = self.code_chunker.chunk(css_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(Chunker.consolidate_chunks_into_file(chunks))
        self.assertEqual(num_lines, len(css_code.split("\n")))
        self.assertIn(css_code, final_code)

# Ruby Test Class
class TestCodeChunkerRuby(unittest.TestCase):

    def setUp(self):
        self.patcher = patch('agentless.retreival.TextChunker.count_tokens', side_effect=mock_count_tokens)
        self.mock_count_tokens = self.patcher.start()
        self.code_chunker = CodeChunker(file_extension='rb')
        self.mock_codebase = mock_codebase

    def tearDown(self):
        self.patcher.stop()

    def test_chunk_ruby_code(self):
        rb_code = self.mock_codebase['example.rb']
        chunks = self.code_chunker.chunk(rb_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(final_code)
        self.assertEqual(num_lines, len(rb_code.split("\n")))
        self.assertIn(rb_code, final_code)
        self.assertGreater(len(chunks), 1)  # Ensure the code is actually chunked

# PHP Test Class
class TestCodeChunkerPHP(unittest.TestCase):

    def setUp(self):
        self.patcher = patch('agentless.retreival.TextChunker.count_tokens', side_effect=mock_count_tokens)
        self.mock_count_tokens = self.patcher.start()
        self.code_chunker = CodeChunker(file_extension='php')
        self.mock_codebase = mock_codebase

    def tearDown(self):
        self.patcher.stop()

    def test_chunk_php_code(self):
        php_code = self.mock_codebase['example.php']
        chunks = self.code_chunker.chunk(php_code, token_limit=20)
        Chunker.print_chunks(chunks)
        final_code = Chunker.consolidate_chunks_into_file(chunks)
        num_lines = Chunker.count_lines(final_code)
        self.assertEqual(num_lines, len(php_code.split("\n")))
        self.assertIn(php_code, final_code)
        self.assertGreater(len(chunks), 1)  # Ensure the code is actually chunked


if __name__ == '__main__':
    unittest.main()