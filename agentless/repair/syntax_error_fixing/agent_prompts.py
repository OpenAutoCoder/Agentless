def syntax_error_fixing_w_diffs_prompt(syntax_error, code_file_with_lines, format_instructions):
    return f"""
                    
    There is a syntax error in a code file that you need to fix. The syntax error is:
    {syntax_error}.
    
    Here is the code file with the syntax error, with the line numbers included:
    {code_file_with_lines}
    
    When looking at the syntax error, make sure you consider the logic of the code. For example, if a closing parenthesis is missing, make sure you add it in the right place to ensure the code still works as intended.
    
    You job is to create a diff file that contains the changes needed to fix the syntax error in the code file.

    {format_instructions}
                    """

diff_file_format_instructions = """
Format the response as a markdown file containing the diff. The diff should be in the unified diff format, which looks like this:
  
```diff
--- a/file_to_work_on
+++ b/file_to_work_on
@@ -23,1 +23,1 @@
-old line
+new line
@@ -30,2 +30,2 @@
-old line 1
-old line 2
+new line 1
+new line 2
@@ -40,3 +40,4 @@
-old line 1
-old line 2
-old line 3
+new line 1
+new line 2
+new line 3
+new line 4
@@ -49,0 +50,3 @@ 
+new line 1
+new line 2
+new line 3
@@ -59,1 +63,0 @@  
-old line
@@ -69,4 +72,4 @@  
 context line 1
 context line 2
-old line 1
+new line 1
 context line 3
 context line 4
```

You must adhere to the following rules when creating the diff:
1. Order Changes Sequentially: Start from the earliest line number to the latest, creating hunks for each change.
2. Format Each Hunk: For each change, write the hunk header to specify where the change occurs, followed by the changed lines. Use - for removals and + for additions.
3. When multiple consecutive lines are changed, group them into a single hunk.
4. Provide context by including unchanged lines around modifications when possible. This makes the diff easier to read and apply.

If no changes are needed, simply return:
"No changes needed"
"""
