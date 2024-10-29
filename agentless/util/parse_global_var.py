import libcst as cst
import libcst.matchers as m
from libcst.display import dump


class GlobalVariableVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self):
        self.global_assigns = []

    def leave_Module(self, original_node: cst.Module) -> list:
        assigns = []
        for stmt in original_node.body:
            if m.matches(stmt, m.SimpleStatementLine()) and m.matches(
                stmt.body[0], m.Assign()
            ):
                start_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).start
                end_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).end
                assigns.append([stmt, start_pos, end_pos])
        self.global_assigns.extend(assigns)


def parse_global_var_from_code(file_content: str) -> dict[str, dict]:
    """Parse global variables."""
    try:
        tree = cst.parse_module(file_content)
    except:
        return file_content

    wrapper = cst.metadata.MetadataWrapper(tree)
    visitor = GlobalVariableVisitor()
    wrapper.visit(visitor)

    global_assigns = {}
    for assign_stmt, start_pos, end_pos in visitor.global_assigns:
        for t in assign_stmt.body:
            try:
                targets = [t.targets[0].target.value]
            except:
                try:
                    targets = t.targets[0].target.elements
                    targets = [x.value.value for x in targets]
                except:
                    targets = []
            for target_var in targets:
                global_assigns[target_var] = {
                    "start_line": start_pos.line,
                    "end_line": end_pos.line,
                }
    return global_assigns


def test_parse_global_var_from_file():
    code = """
\"\"\"
this is a module
...
\"\"\"
const_var = {1,2,3}
const_dict = {
    'day': 'Monday',
    'month': 'January',
}
a, b = 1, 2
import os

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
    res = parse_global_var_from_code(code)
    assert res == {
        "const_var": {"start_line": 6, "end_line": 6},
        "const_dict": {"start_line": 7, "end_line": 10},
        "a": {"start_line": 11, "end_line": 11},
        "b": {"start_line": 11, "end_line": 11},
    }


if __name__ == "__main__":
    test_parse_global_var_from_file()
