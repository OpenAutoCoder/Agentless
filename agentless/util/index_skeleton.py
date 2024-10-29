import libcst as cst
import libcst.matchers as m


class GlobalVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self):
        # includes both imports and assigns
        self.global_stmt = []
        self.global_imports = []

    def leave_Module(self, original_node: cst.Module) -> list:
        for stmt in original_node.body:
            if m.matches(stmt, m.SimpleStatementLine()) and (
                m.matches(stmt.body[0], m.Assign())
            ):
                expr = cst.parse_module("").code_for_node(stmt).strip()
                self.global_stmt.append(expr)

            if m.matches(stmt, m.SimpleStatementLine()) and (
                m.matches(stmt.body[0], m.Import())
                or m.matches(stmt.body[0], m.ImportFrom())
            ):
                expr = cst.parse_module("").code_for_node(stmt).strip()
                self.global_imports.append(expr)


def parse_global_stmt_from_code(file_content: str) -> tuple[str, str]:
    """Parse global variables."""
    try:
        tree = cst.parse_module(file_content)
        wrapper = cst.metadata.MetadataWrapper(tree)
        visitor = GlobalVisitor()
        wrapper.visit(visitor)

        return "\n".join(visitor.global_stmt), "\n".join(visitor.global_imports)
    except:
        return "", ""


code = """
\"\"\"
this is a module
...
\"\"\"
const = {1,2,3}
import os
from ds import get
from ds import *
class fooClass:
    '''this is a class'''

    def __init__(self, x):
        '''initialization.'''
        self.x = x

    def print(self):
        print(self.x)

def test():
    a = fooClass(3)
    a.print()
"""


def test_parse():
    global_stmt = parse_global_stmt_from_code(code)
    print(global_stmt)


if __name__ == "__main__":
    test_parse()
