import pandas as pd
import logging
from typing import Generator, List


class DataFrameChunker:
    """
    Chunker class for splitting pandas dataframes.
    
    dataframe: (pd.DataFrame) dataframe to be split
    n_approx: (int) approximate row number in each chunk
    column_name: (str) name of the column to chunk along
    """
    def __init__(self, dataframe: pd.DataFrame, n_approx: int, column_name: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataframe = dataframe
        self.n_approx = n_approx
        self.buffer = []
        self.column_name = column_name        
        
        self.validate_params()


    def validate_params(self):
        """Validation of the DataFrameChunker parameters."""
        if self.column_name not in self.dataframe.columns:
            self.logger.error(f"Column '{self.column_name}' not found in DataFrame.")
            raise KeyError(f"Column '{self.column_name}' not found in DataFrame.")
        
        if not isinstance(self.n_approx, int) or self.n_approx <= 0:
            self.logger.error(f"Invalid value for n_approx: {self.n_approx}. It must be a positive integer.")
            raise ValueError(f"n_approx must be a positive integer, got {self.n_approx}.")
        
        if self.dataframe.empty:
            self.logger.warning("The provided DataFrame is empty. No chunks will be created.")
        
        self.logger.info("DataFrameChunker initialized successfully.")
        return True


    def add_to_buffer(self, chunk: pd.DataFrame):
        """Store a chunk temporarily in the chunk cache."""
        self.buffer.append(chunk)
        self.logger.debug(f"Stored a chunk, cache now contains {len(self.buffer)} chunks")


    def flush_buffer(self) -> List[pd.DataFrame]:
        """Clear the chunk cache and return its content."""
        self.logger.info("Clearing the chunk cache")
        cached_data = self.buffer[:]
        self.buffer.clear()
        return cached_data


    def chunk(self) -> Generator[pd.DataFrame, None, None]:
        """Generator that splits the DataFrame into chunks and utilizes cache."""
        try:
            self.logger.info(f"Splitting DataFrame into chunks of approximately {self.n_approx} rows")
            groups = self.dataframe.groupby(self.column_name).size().to_dict()

            current_chunk = []

            for key, size in groups.items():
                current_chunk.extend([key] * size)
                if len(current_chunk) >= self.n_approx:
                    chunk_df = self.dataframe[self.dataframe[self.column_name].isin(current_chunk)]
                    self.add_to_buffer(chunk_df)
                    current_chunk = []

            if current_chunk:
                chunk_df = self.dataframe[self.dataframe[self.column_name].isin(current_chunk)]
                self.add_to_buffer(chunk_df)

            for chunk in self.flush_buffer():
                self.logger.debug(f"Yielding a chunk with {len(chunk)} rows")
                yield chunk

            self.logger.info("DataFrame successfully split into chunks")

        except KeyError as e:
            self.logger.error(f"Error: Group-by column '{e}' not found in the DataFrame")
            raise e


if __name__ == "__main__":
    dfs = [
        "2023-01-01 00:00:01",
        "2023-01-01 00:00:01",
        "2023-01-01 00:00:02",
        "2023-01-01 00:00:02",
        "2023-01-01 00:00:02",
        "2023-01-01 00:00:03",
    ]
    df = pd.DataFrame({"dt": dfs})
    
    chunker = DataFrameChunker(df.head(10), n_approx=2, column_name="dt")

    for i, chunk in enumerate(chunker.chunk()):
        print(f"Chunk {i+1}:\n{chunk}\n")
