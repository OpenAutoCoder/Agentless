import unittest

from agentless.util.compress_file import compress_assign_stmts, get_skeleton
from agentless.util.parse_global_var import (
    parse_global_var_from_code,
    test_parse_global_var_from_file,
)


class AnnotatedGlobalsTests(unittest.TestCase):
    def test_skeleton_keeps_annotated_global(self):
        code = "LIMIT: int = 64\n\ndef check(n):\n    return n < LIMIT\n"
        skeleton = get_skeleton(code)
        self.assertIn("LIMIT: int = 64", skeleton)
        self.assertIn("def check(n):", skeleton)
        self.assertNotIn("return n < LIMIT", skeleton)

    def test_skeleton_keeps_annotation_only(self):
        self.assertIn("LIMIT: int", get_skeleton("LIMIT: int\n"))

    def test_skeleton_respects_keep_constant_false(self):
        code = "PLAIN = 1\nLIMIT: int = 64\nDECLARED: str\n"
        self.assertEqual(get_skeleton(code, keep_constant=False).strip(), "")

    def test_global_locations_include_annotated_name(self):
        self.assertEqual(
            parse_global_var_from_code("# heading\nLIMIT: int = 64\n"),
            {"LIMIT": {"start_line": 2, "end_line": 2}},
        )

    def test_global_locations_include_annotation_only(self):
        self.assertEqual(
            parse_global_var_from_code("DECLARED: str\n"),
            {"DECLARED": {"start_line": 1, "end_line": 1}},
        )

    def test_multiline_annotated_location(self):
        code = "VALUES: list[int] = [\n    1,\n    2,\n]\n"
        self.assertEqual(
            parse_global_var_from_code(code),
            {"VALUES": {"start_line": 1, "end_line": 4}},
        )

    def test_local_and_class_annotations_are_not_globals(self):
        code = (
            "class C:\n    field: int = 1\n\n"
            "def f():\n    local: int = 2\n\nobj.field: int = 3\n"
        )
        self.assertEqual(parse_global_var_from_code(code), {})

    def test_existing_plain_assignments(self):
        test_parse_global_var_from_file()

    def test_large_annotated_assignment_is_compressed(self):
        code = "VALUES: list[int] = [\n" + "    1,\n" * 40 + "]\n"
        compressed = compress_assign_stmts(
            code, total_lines=10, prefix_lines=3, suffix_lines=3
        )
        self.assertIn("VALUES: list[int] = [", compressed)
        self.assertIn("...\n", compressed)
        self.assertLess(len(compressed.splitlines()), 15)

    def test_large_plain_assignment_still_compressed(self):
        code = "VALUES = [\n" + "    1,\n" * 40 + "]\n"
        self.assertIn("...\n", compress_assign_stmts(code, total_lines=10))

    def test_small_annotated_assignment_is_unchanged(self):
        code = "VALUES: list[int] = [1, 2]\n"
        self.assertEqual(compress_assign_stmts(code), code)

    def test_variable_location_reaches_original_source(self):
        from agentless.util.preprocess_data import transfer_arb_locs_to_locs

        code = "# heading\nLIMIT: int = 64\n\ndef check(n):\n    return n < LIMIT\n"
        locations, context = transfer_arb_locs_to_locs(
            ["variable: LIMIT"],
            None,
            "limit.py",
            context_window=0,
            file_content=code,
        )
        self.assertEqual(locations, [(2, 2)])
        self.assertEqual(context, [(2, 2)])


if __name__ == "__main__":
    unittest.main()
