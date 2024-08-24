import re
from pydantic import BaseModel, Field, field_validator

class DiffFileContents(BaseModel):
    diff: str = Field(..., description="The entire diff file with the changes applied or 'No changes needed'.")

    @field_validator('diff')
    def validate_diff_content(cls, v):
        if v != "No changes needed":
            placeholder_patterns = [
                r"rest\s+of\s+code",
                r"implement\s+logic"
            ]
            placeholder_regex = re.compile(r'|'.join(placeholder_patterns), re.IGNORECASE)
            if placeholder_regex.search(v):
                raise ValueError(f"Diff content must not contain incomplete phrases like '{placeholder_regex.pattern}'")
            
            if not v.endswith('\n'):
                v += '\n'
        return v

    @classmethod
    def extract_diff_content(cls, text: str):
        diff_block_start_pattern = r"```diff" or r"```" or r"text=" or r"text='"
        diff_block_end_pattern = "```" or ""
        
        start_match = re.search(diff_block_start_pattern, text)
        if start_match:
            end_index = text.find(diff_block_end_pattern, start_match.end())
            if end_index != -1:
                content = text[start_match.end():end_index].strip()
                return cls(diff=content)
            else:
                raise ValueError("The closing delimiter for the diff code block was not found.")
        else:
            if "No changes needed" in text:
                return cls(diff="No changes needed")
            else:
                raise ValueError("No valid diff content or 'No changes needed' phrase found within a Markdown code block.")

