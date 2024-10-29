import re

import libcst as cst
import libcst.matchers as m


class CompressTransformer(cst.CSTTransformer):
    DESCRIPTION = str = "Replaces function body with ..."
    replacement_string = '"__FUNC_BODY_REPLACEMENT_STRING__"'

    def __init__(self, keep_constant=True, keep_indent=False):
        self.keep_constant = keep_constant
        self.keep_indent = keep_indent

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        new_body = [
            stmt
            for stmt in updated_node.body
            if m.matches(stmt, m.ClassDef())
            or m.matches(stmt, m.FunctionDef())
            or (
                self.keep_constant
                and m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Assign())
            )
        ]
        return updated_node.with_changes(body=new_body)

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        # Remove docstring in the class body
        new_body = [
            stmt
            for stmt in updated_node.body.body
            if not (
                m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Expr())
                and m.matches(stmt.body[0].value, m.SimpleString())
            )
        ]
        return updated_node.with_changes(body=cst.IndentedBlock(body=new_body))

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        if not self.keep_indent:
            # replace with unindented statement
            new_expr = cst.Expr(value=cst.SimpleString(value=self.replacement_string))
            new_body = cst.IndentedBlock((new_expr,))
            return updated_node.with_changes(body=new_body)
        else:
            # replace with indented statement
            # new_expr = [cst.Pass()]
            new_expr = [
                cst.Expr(value=cst.SimpleString(value=self.replacement_string)),
            ]
            return updated_node.with_changes(
                body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=new_expr)])
            )


class GlobalVariableVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self):
        self.assigns = []

    def leave_Assign(self, original_node: cst.Module) -> list:
        stmt = original_node
        start_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).start
        end_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).end
        self.assigns.append([stmt, start_pos, end_pos])


code = """
\"\"\"
this is a module
...
\"\"\"
const = {1,2,3}
import os

class fooClass:
    '''this is a class'''

    def __init__(self, x):
        '''initialization.'''
        self.x = x

    def print(self):
        print(self.x)

large_var = {
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21,
}

def test():
    a = fooClass(3)
    a.print()

"""


def remove_lines(raw_code, remove_line_intervals):
    # TODO: speed up this function
    # remove_line_intervals.sort()

    # Remove lines
    new_code = ""
    for i, line in enumerate(raw_code.splitlines()):
        # intervals are one-based
        if not any(start <= i + 1 <= end for start, end in remove_line_intervals):
            new_code += line + "\n"
        if any(start == i + 1 for start, _ in remove_line_intervals):
            new_code += "...\n"
    return new_code


def compress_assign_stmts(raw_code, total_lines=30, prefix_lines=10, suffix_lines=10):
    try:
        tree = cst.parse_module(raw_code)
    except Exception as e:
        print(e.__class__.__name__, e)
        return raw_code

    wrapper = cst.metadata.MetadataWrapper(tree)
    visitor = GlobalVariableVisitor()
    wrapper.visit(visitor)

    remove_line_intervals = []
    for stmt in visitor.assigns:
        if stmt[2].line - stmt[1].line > total_lines:
            remove_line_intervals.append(
                (stmt[1].line + prefix_lines, stmt[2].line - suffix_lines)
            )
    return remove_lines(raw_code, remove_line_intervals)


def get_skeleton(
    raw_code,
    keep_constant: bool = True,
    keep_indent: bool = False,
    compress_assign: bool = False,
    total_lines=30,
    prefix_lines=10,
    suffix_lines=10,
):
    try:
        tree = cst.parse_module(raw_code)
    except:
        return raw_code

    transformer = CompressTransformer(keep_constant=keep_constant, keep_indent=True)
    modified_tree = tree.visit(transformer)
    code = modified_tree.code

    if compress_assign:
        code = compress_assign_stmts(
            code,
            total_lines=total_lines,
            prefix_lines=prefix_lines,
            suffix_lines=suffix_lines,
        )

    if keep_indent:
        code = code.replace(CompressTransformer.replacement_string + "\n", "...\n")
        code = code.replace(CompressTransformer.replacement_string, "...\n")
    else:
        pattern = f"\\n[ \\t]*{CompressTransformer.replacement_string}"
        replacement = "\n..."
        code = re.sub(pattern, replacement, code)

    return code


def test_compress():
    skeleton = get_skeleton(code, True)
    print(skeleton)


def test_compress_var():
    print("LOC: ", len(code.split("\n")))
    skeleton = get_skeleton(
        code,
        True,
        keep_indent=False,
        compress_assign=True,
        total_lines=10,
        prefix_lines=5,
        suffix_lines=5,
    )
    print(skeleton)
    print("LOC: ", len(skeleton.split("\n")))


if __name__ == "__main__":
    # test_compress()
    test_compress_var()
