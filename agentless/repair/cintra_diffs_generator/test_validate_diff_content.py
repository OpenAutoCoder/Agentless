import unittest
from app.util.CodeChanging.InMemoryDiffApplier import InMemoryDiffApplier
from app.util.CodeChanging.code_changing_models import DiffFileContents

class TestDiffFileContents(unittest.TestCase):

    def test_extract_diff_content_with_valid_diff(self):
        text = """
```diff
--- a/file_to_work_on
+++ b/file_to_work_on
@@ -1,4 +1,4 @@
-The original line
+The new line
```
"""
        expected_result = """
--- a/file_to_work_on
+++ b/file_to_work_on
@@ -1,4 +1,4 @@
-The original line
+The new line
""".strip()
        result = DiffFileContents.extract_diff_content(text)
        self.assertEqual(result.diff.strip(), expected_result)

    def test_extract_diff_content_with_leading_text(self):
        text = """
Here's the diff file
```diff
--- a/file_to_work_on
+++ b/file_to_work_on
@@ -1,4 +1,4 @@
-The original line
+The new line
```
"""
        expected_result = """
--- a/file_to_work_on
+++ b/file_to_work_on
@@ -1,4 +1,4 @@
-The original line
+The new line
""".strip()
        result = DiffFileContents.extract_diff_content(text)
        self.assertEqual(result.diff.strip(), expected_result)


    def test_extract_diff_content_with_trailing_text(self):
        text = """
```diff
--- a/file_to_work_on
+++ b/file_to_work_on
@@ -1,4 +1,4 @@
-The original line
+The new line
```
you can insert this into your new code file
"""
        expected_result = """
--- a/file_to_work_on
+++ b/file_to_work_on
@@ -1,4 +1,4 @@
-The original line
+The new line
""".strip()
        result = DiffFileContents.extract_diff_content(text)
        self.assertEqual(result.diff.strip(), expected_result)

    def test_extract_diff_content_with_leading_and_trailing_text(self):
        text = """
Here's the diff file
```diff
--- a/file_to_work_on
+++ b/file_to_work_on
@@ -1,4 +1,4 @@
-The original line
+The new line
```
you can insert this into your new code file
"""
        expected_result = """
--- a/file_to_work_on
+++ b/file_to_work_on
@@ -1,4 +1,4 @@
-The original line
+The new line
""".strip()
        result = DiffFileContents.extract_diff_content(text)
        self.assertEqual(result.diff.strip(), expected_result)

    def test_incomplete_code(self):
        text = """
        ```diff
--- a/file_to_work_on
+++ b/file_to_work_on
@@ -1,4 +1,4 @@
-The original line
+The new line
//rest of code
```
"""
        with self.assertRaises(ValueError) as context:
            DiffFileContents.extract_diff_content(text)
        self.assertTrue("Diff content must not contain incomplete phrases like 'rest\\s+of\\s+code|implement\\s+logic'" in str(context.exception))
       

    def test_extract_diff_content_with_no_changes_needed(self):
        text = "No changes needed"
        result = DiffFileContents.extract_diff_content(text)
        self.assertEqual(result.diff, "No changes needed")

    def test_extract_diff_content_with_invalid_diff(self):
        text = """
        Not a valid diff block:
        Random text here
        """
        with self.assertRaises(ValueError):
            DiffFileContents.extract_diff_content(text)

    def test_extract_diff_content_missing_end_delimiter(self):
        text = """
        ```diff
        --- a/file_to_work_on
        +++ b/file_to_work_on
        @@ -1,4 +1,4 @@
        -The original line
        +The new line
        """
        with self.assertRaises(ValueError):
            DiffFileContents.extract_diff_content(text)

class TestInMemoryDiffApplier(unittest.TestCase):
    
    def test_apply_diff(self):
        original_content = 'print("Hello, world!")\n'
        diff_content = '--- a\n+++ b\n@@ -1 +1 @@\n-print("Hello, world!")\n+print("Hello, Python!")\n'
        applier = InMemoryDiffApplier(original_content, diff_content)
        result = applier.apply_diff()
        self.assertEqual(result['updated_content'], 'print("Hello, Python!")\n')

    def test_apply_diff_with_invalid_diff(self):
        original_content = 'print("Hello, world!")\n'
        diff_content = 'This is not a valid diff'
        applier = InMemoryDiffApplier(original_content, diff_content)
        with self.assertRaises(ValueError):
            applier.apply_diff()

    def test_apply_diff_with_multiple_lines(self):
        original_content = 'print("Hello, world!")\nprint("Goodbye, world!")\n'
        diff_content = '--- a\n+++ b\n@@ -1,2 +1,2 @@\n-print("Hello, world!")\n+print("Hello, Python!")\n-print("Goodbye, world!")\n+print("Goodbye, Python!")\n'
        applier = InMemoryDiffApplier(original_content, diff_content)
        result = applier.apply_diff()
        self.assertEqual(result['updated_content'], 'print("Hello, Python!")\nprint("Goodbye, Python!")\n')

# If this script is run directly, execute the tests
if __name__ == '__main__':
    unittest.main()