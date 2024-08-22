import os
import subprocess
import tempfile
import re

class InMemoryDiffApplier:
    """
    A class to apply in-memory diffs to file contents. This class handles the application of diffs,
    correction of malformed patches, and reordering of misordered hunks.
    """

    def __init__(self, original_file_content: str, diff_content: str):
        """
        Initialize the InMemoryDiffApplier with the original file content and the diff content.

        :param original_file_content: The original content of the file as a string.
        :param diff_content: The diff content to be applied to the original file content as a string.
        """
        self.original_file_content = original_file_content
        self.diff_content = diff_content
        self.corrections_made = False  # Track if any corrections are made

    def apply_diff(self):
        """
        Apply the diff to the original file content. If the diff is malformed or contains misordered hunks,
        attempt to correct and reapply the diff.

        :return: A dictionary containing the updated content and a flag indicating if corrections were made.
        :raises ValueError: If the diff cannot be applied after the maximum number of attempts.
        """
        max_attempts = 1
        for attempt in range(max_attempts):
            try:
                updated_content = self._attempt_to_apply_diff()
                if self.corrections_made:
                    print("Successfully fixed diff file")
                return {
                    "updated_content": updated_content,
                    "corrections_made": self.corrections_made
                }
            except ValueError as e:
                print(f"Correction attempt: {attempt}")
                error_message = str(e)
                if "malformed patch" in error_message and attempt < max_attempts - 1:
                    print("Error Found In Diff File:", e)
                    print(f'diff file with error {self.diff_content}')
                    line_with_error = int(re.search(r"malformed patch at line (\d+)", error_message).group(1))
                    self.diff_content = self._correct_hunk_headers_based_on_content(line_with_error)
                    self.corrections_made = True
                elif "misordered hunks" in error_message and attempt < max_attempts - 1:
                    self.diff_content = self.reorder_diff_hunks(self.diff_content)
                    self.corrections_made = True
                elif attempt == max_attempts - 1:
                    raise ValueError(f"Failed to apply diff after {attempt + 1} attempts. Last error: {error_message}")
                else:
                    print(f"Other error not solvable: ", e)
                    pass
        raise ValueError(f"Failed to apply diff after {max_attempts} attempts.")

    def _attempt_to_apply_diff(self) -> str:
        """
        Attempt to apply the diff to the original file content using the `patch` command.

        :return: The updated file content as a string.
        :raises ValueError: If the diff cannot be applied.
        """
        with tempfile.NamedTemporaryFile(delete=False) as original_tmp, \
             tempfile.NamedTemporaryFile(delete=False) as diff_tmp:
            original_tmp_path = original_tmp.name
            diff_tmp_path = diff_tmp.name
            original_tmp.write(self.original_file_content.encode())
            diff_tmp.write(self.diff_content.encode())
            original_tmp.flush()
            diff_tmp.flush()

        command = ['patch', '--force', '--ignore-whitespace', original_tmp_path, diff_tmp_path]

        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise ValueError(f"Failed to apply diff. Return code: {result.returncode}. STDOUT: {result.stdout}, STDERR: {result.stderr}")
        else:
            print(f"Diff applied successfully. STDOUT: {result.stdout}")

        with open(original_tmp_path, 'r') as updated_file:
            updated_content = updated_file.read()

        os.remove(original_tmp_path)
        os.remove(diff_tmp_path)

        return updated_content

    def _correct_hunk_headers_based_on_content(self, line_with_error: int = None) -> str:
        """
        Attempt to correct hunk headers in the diff content based on the original file content.

        :param line_with_error: The line number where the error occurred.
        :return: The corrected diff content as a string.
        :raises ValueError: If the original starting line for the hunk cannot be found.
        """
        print("Attempting to fix hunk headers for line:", line_with_error)
        try:
            original_lines = self.original_file_content.split('\n')
            diff_lines = self.diff_content.split('\n')
            new_diff_content = []
            new_lines_changed_before_error = []
            i = 0
            while i < len(diff_lines):
                line = diff_lines[i]
                if i == line_with_error - 1:
                    print("Found the line with error, starting correction process.")
                    hunk_header = line
                    hunk_body = []
                    i += 1
                    while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
                        hunk_body.append(diff_lines[i])
                        i += 1

                    # search for context lines or subtracted lines in the hunk body and extract existing content in the hunk
                    existing_content_in_hunk = []
                    for line in hunk_body:
                        # if the line starts with a '-' or is not a '+' line, it is a context line or a subtracted line. This means it is part of the original file
                        if line.startswith('-') or not line.startswith('+'):
                            existing_content_in_hunk.append(original_lines[int(line[1:].split(',')[0]) - 1])

                    # find the correct original starting line for the hunk by searching for the existing content in the original file
                    original_start_line = None
                    # go through each line of the existing content in the hunk and when you find a match, go to the next line of the existing content in the hunk and see if the next line matches it. Do this until we've exhaused the existing content in the hunk. If this happens, then the original starting line is the line before the first match
                    for i in range(len(existing_content_in_hunk)):
                        original_start_line = original_lines.index(existing_content_in_hunk[i])
                        for j in range(i + 1, len(existing_content_in_hunk)):
                            if original_lines[original_start_line + j - i] != existing_content_in_hunk[j]:
                                original_start_line = None
                                break
                        if original_start_line is not None:
                            break

                    if original_start_line is None:
                        raise ValueError(f"Failed to find original starting line for hunk: {hunk_header}")

                    # Ensure the correct amount of lines affected in the original file is reflected in the hunk header
                    number_of_lines_affected_in_original_file = len([line for line in hunk_body if line.startswith('-')])
                    hunk_header_parts = hunk_header.split(' ')
                    hunk_header_parts[1] = f"-{original_start_line},{number_of_lines_affected_in_original_file}"

                    # Now look at the starting point in the new file based on prior hunks in the new diff, if any
                    if len(new_diff_content) > 0:
                        last_hunk = new_diff_content[-1]
                        last_hunk_match = re.match(r'^@@ -\d+,\d+ \+(\d+),\d+ @@.*$', last_hunk)
                        if last_hunk_match:
                            starting_line_in_new_file = int(last_hunk_match.group(1)) + sum(len(line.split('\n')) for line in new_lines_changed_before_error)
                            hunk_header_parts[2] = f"+{starting_line_in_new_file},{number_of_lines_affected_in_original_file}"
                    new_diff_content.append(' '.join(hunk_header_parts))
                    new_diff_content.extend(hunk_body)
                    i += 1
                elif diff_lines[i].startswith("@@"):
                    new_diff_content.append(line)
                    # Calculate how many lines were changed before the error
                    if line_with_error is not None:
                        new_lines_changed_before_error.append(line)
                    i += 1
                else:
                    new_diff_content.append(line)
                    i += 1

            # Reassemble the corrected diff content
            corrected_diff_content = '\n'.join(new_diff_content) + '\n'
            print("Correction process completed. Corrected diff content:")
            print(corrected_diff_content)
            return corrected_diff_content
        except Exception as e:
            print(f"An error occurred during the correction process: {e}")
            raise

    def reorder_diff_hunks(self, diff_content: str) -> str:
        """
        Reorders the hunks in a diff file based on the starting line numbers in the original file,
        while preserving non-hunk content. Ensures proper newline handling.

        :param diff_content: The content of the diff file as a string.
        :return: The reordered diff file content as a string.
        """
        print("Reordering diff hunks")
        hunk_header_pattern = re.compile(r'^@@ -(\d+),\d+ \+(\d+),\d+ @@.*$', re.MULTILINE)

        lines = diff_content.split('\n')
        hunks = []
        non_hunk_content = []
        i = 0

        # Variable to hold lines that are not part of a hunk
        interspersed_non_hunk_content = []

        while i < len(lines):
            if lines[i].startswith("@@"):
                match = hunk_header_pattern.match(lines[i])
                if not match:
                    i += 1
                    continue

                original_start_line = int(match.group(1))
                hunk_header = lines[i]
                i += 1

                hunk_body = []
                while i < len(lines) and not lines[i].startswith("@@"):
                    hunk_body.append(lines[i])
                    i += 1

                hunks.append((hunk_header, hunk_body, original_start_line, interspersed_non_hunk_content))
                # Reset the non-hunk content tracker after a hunk has been found
                interspersed_non_hunk_content = []
            else:
                # Collect non-hunk content to be preserved and reinserted appropriately
                interspersed_non_hunk_content.append(lines[i])
                i += 1

        # Sort hunks based on the original start line number
        hunks.sort(key=lambda h: h[2])

        reordered_diff_content = []
        for hunk in hunks:
            # Insert non-hunk content that appears before the current hunk
            reordered_diff_content.extend(hunk[3])

            reordered_diff_content.append(hunk[0])  # Hunk header
            reordered_diff_content.extend(hunk[1])  # Hunk body

        # Ensure any remaining non-hunk content is appended (e.g., content after the last hunk)
        reordered_diff_content.extend(interspersed_non_hunk_content)

        # Ensuring that each line ends with a newline character
        corrected_diff_content = '\n'.join(reordered_diff_content) + '\n'

        return corrected_diff_content