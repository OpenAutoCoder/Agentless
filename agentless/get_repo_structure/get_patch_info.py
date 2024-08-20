import argparse
import json
import re
from collections import defaultdict


def parse_patch(patch):
    """
    Parse a git patch into a structured format.

    Parameters:
        patch (str): The git patch as a string.

    Returns:
        list: A list of dictionaries representing the file changes and hunks.
    """
    file_changes = []
    current_file = None
    current_hunk = None
    deleted_lines = 0

    patch_lines = patch.split("\n")
    for line in patch_lines:
        if line.startswith("diff --git"):
            # Reset for new files
            if current_file:
                file_changes.append(current_file)
            current_file = {"file": "", "hunks": []}
        elif line.startswith("--- a/"):
            pass
        elif line.startswith("+++ b/"):
            if current_file is not None:
                current_file["file"] = line[6:]
        elif line.startswith("@@ "):
            if current_file is not None:
                match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if match:
                    current_hunk = {"start_line": int(match.group(2)), "changes": []}
                    current_file["hunks"].append(current_hunk)
                    deleted_lines = 0
                    added_lines = 0
        elif line.startswith("+") or line.startswith("-"):
            if current_hunk is not None:
                change_type = "add" if line.startswith("+") else "delete"
                if change_type == "delete":
                    deleted_lines += 1
                    current_hunk["changes"].append(
                        {
                            "type": change_type,
                            "content": line[1:].strip(),
                            "line": current_hunk["start_line"] - added_lines,
                        }
                    )
                    current_hunk["start_line"] += 1
                else:
                    added_lines += 1
                    current_hunk["changes"].append(
                        {
                            "type": change_type,
                            "content": line[1:].strip(),
                            "line": current_hunk["start_line"] - deleted_lines,
                        }
                    )
                    current_hunk["start_line"] += 1
        else:
            if current_hunk is not None:
                current_hunk["start_line"] += 1

    if current_file:
        file_changes.append(current_file)

    return file_changes
