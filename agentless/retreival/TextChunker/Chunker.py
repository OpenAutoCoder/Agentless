from abc import ABC, abstractmethod
from agentless.retreival.TextChunker.CodeParser import CodeParser
from agentless.retreival.TextChunker.count_tokens import count_tokens

class Chunker(ABC):
    def __init__(self, encoding_name="gpt-4"):
        self.encoding_name = encoding_name

    @abstractmethod
    def chunk(self, content, token_limit):
        pass

    @abstractmethod
    def get_chunk(self, chunked_content, chunk_number):
        pass

    @staticmethod
    def print_chunks(chunks):
        for chunk_number, chunk_code in chunks.items():
            print(f"Chunk {chunk_number}:")
            print("="*40)
            print(chunk_code)
            print("="*40)

    @staticmethod
    def consolidate_chunks_into_file(chunks):
        return "\n".join(chunks.values())
    
    @staticmethod
    def count_lines(consolidated_chunks):
        lines = consolidated_chunks.split("\n")
        return len(lines)

class CodeChunker(Chunker):
    def __init__(self, file_extension, encoding_name="gpt-4"):
        super().__init__(encoding_name)
        self.file_extension = file_extension
        try:
            self.code_parser = CodeParser(self.file_extension)
        except ValueError:
            print(f"Warning: Parser for extension '{self.file_extension}' not available. Using ChunkDictionary instead.")
            self.code_parser = ChunkDictionary()

    def chunk(self, code, token_limit) -> dict:
        """
        This function takes a code and chunks it into smaller chunks, where each chunk has a total number
        of tokens less than or equal to the token_limit.
        """
        chunks = {}
        current_chunk = ""
        token_count = 0
        lines = code.split("\n")
        i = 0
        chunk_number = 1
        start_line = 0
        breakpoints = sorted(self.code_parser.get_lines_for_points_of_interest(code, self.file_extension))
        comments = sorted(self.code_parser.get_lines_for_comments(code, self.file_extension))
        adjusted_breakpoints = []
        for bp in breakpoints:
            current_line = bp - 1
            highest_comment_line = None  # Initialize with None to indicate no comment line has been found yet
            while current_line in comments:
                highest_comment_line = current_line  # Update highest comment line found
                current_line -= 1  # Move to the previous line

            if highest_comment_line:  # If a highest comment line exists, add it
                adjusted_breakpoints.append(highest_comment_line)
            else:
                adjusted_breakpoints.append(bp)  # If no comments were found before the breakpoint, add the original breakpoint

        breakpoints = sorted(set(adjusted_breakpoints))  # Ensure breakpoints are unique and sorted
        
        while i < len(lines):
            line = lines[i]
            new_token_count = count_tokens(line, self.encoding_name)
            if token_count + new_token_count > token_limit:
                
                # Set the stop line to the last breakpoint before the current line
                if i in breakpoints:
                    stop_line = i
                else:
                    stop_line = max(max([x for x in breakpoints if x < i], default=start_line), start_line)

                # If the stop line is the same as the start line, it means we haven't reached a breakpoint yet and we need to move to the next line to find one
                if stop_line == start_line and i not in breakpoints:
                    token_count += new_token_count
                    i += 1
                
                # If the stop line is the same as the start line and the current line is a breakpoint, it means we can create a chunk with just the current line
                elif stop_line == start_line and i == stop_line:
                    token_count += new_token_count
                    i += 1
                
                
                # If the stop line is the same as the start line and the current line is a breakpoint, it means we can create a chunk with just the current line
                elif stop_line == start_line and i in breakpoints:
                    current_chunk = "\n".join(lines[start_line:stop_line])
                    if current_chunk.strip():  # If the current chunk is not just whitespace
                        chunks[chunk_number] = current_chunk  # Using chunk_number as key
                        chunk_number += 1
                       
                    token_count = 0
                    start_line = i
                    i += 1

                # If the stop line is different from the start line, it means we're at the end of a block
                else:
                    current_chunk = "\n".join(lines[start_line:stop_line])
                    if current_chunk.strip():
                        chunks[chunk_number] = current_chunk  # Using chunk_number as key
                        chunk_number += 1
                       
                    i = stop_line
                    token_count = 0
                    start_line = stop_line
            else:
                # If the token count is still within the limit, add the line to the current chunk
                token_count += new_token_count
                i += 1

        # Append remaining code, if any, ensuring it's not empty or whitespace
        current_chunk_code = "\n".join(lines[start_line:])
        if current_chunk_code.strip():  # Checks if the chunk is not just whitespace
            chunks[chunk_number] = current_chunk_code  # Using chunk_number as key
            
        return chunks

    def get_chunk(self, chunked_codebase, chunk_number):
        return chunked_codebase[chunk_number]

class ChunkDictionary(Chunker):
    def __init__(self, encoding_name="gpt-4"):
        super().__init__(encoding_name)

    def chunk(self, dictionary, chunk_token_size):
        """
        This function takes a dictionary and chunks it into smaller dictionaries, where each dictionary has a total number
        of tokens less than or equal to the chunk_token_size. This is useful for chunking a text file into smaller pieces.
        :param json_object:
        :param chunk_token_size:
        :param encoding_name:
        :return:
        """

        output_dict = {}
        chunk = {}
        tokens_in_chunk = 0

        for key, value in dictionary.items():

            tokens_in_item = count_tokens(str(value), self.encoding_name)
            if tokens_in_item > chunk_token_size:
                print(f"Ignoring item {key} because it's too large ({tokens_in_item} tokens)")
                continue
            if tokens_in_chunk + tokens_in_item <= chunk_token_size:
                # If the item fits into the current chunk, add it
                chunk[key] = value
                tokens_in_chunk += tokens_in_item
            else:
                # If the item doesn't fit, finalize the current chunk and start a new one
                output_dict[len(output_dict)] = chunk
                chunk = {key: value}
                tokens_in_chunk = tokens_in_item

        # Don't forget the last chunk
        if chunk:
            output_dict[len(output_dict)] = chunk

        # print(f"Chunked codebase into {len(output_dict)} chunks")
        return output_dict

    def get_chunk(self, chunked_content, chunk_number):
        return chunked_content[chunk_number]


class TextStringChunker:
    def __init__(self, chunk_size, encoding_name):
        self.chunk_size = chunk_size
        self.encoding_name = encoding_name

    def chunk(self, input_string):
        chunks = []
        chunk = ""
        tokens_in_chunk = 0
        for line in input_string.split('\n'):
            tokens_in_line = count_tokens(line, self.encoding_name)
            if tokens_in_chunk + tokens_in_line <= self.chunk_size:
                # If the line fits into the current chunk, add it
                chunk += line + '\n'
                tokens_in_chunk += tokens_in_line
            else:
                # If the line doesn't fit, finalize the current chunk and start a new one
                chunks.append(chunk.strip())
                chunk = line
                tokens_in_chunk = tokens_in_line
        # Don't forget the last chunk
        if chunk:
            chunks.append(chunk.strip())
        return chunks