import pandas as pd
import pytest
from df_chunker import DataFrameChunker


@pytest.fixture
def sample_dataframe():
    dfs = pd.date_range(
        "2023-01-01 00:00:00", 
        "2023-01-01 00:00:05", 
        freq="s"
    )
    df = pd.DataFrame({"dt": dfs.repeat(3)})
    return df.head(10)

@pytest.fixture
def empty_dataframe():
    return pd.DataFrame(columns=["dt"])


def test_DataFrameChunker_initialization(sample_dataframe):
    chunker = DataFrameChunker(
        sample_dataframe, 
        n_approx=2, 
        column_name="dt"
    )

    assert chunker.n_approx == 2
    assert chunker.column_name == "dt"
    assert chunker.buffer == []
    assert chunker.validate_params()


def test_invalid_column_name(sample_dataframe):
    with pytest.raises(KeyError):
        DataFrameChunker(sample_dataframe, n_approx=2, column_name="invalid_column")


def test_invalid_n_approx(sample_dataframe):
    with pytest.raises(ValueError):
        DataFrameChunker(sample_dataframe, n_approx=-1, column_name="dt")


def test_add_to_buffer(sample_dataframe):
    chunker = DataFrameChunker(sample_dataframe, n_approx=2, column_name="dt")
    chunk = sample_dataframe.iloc[:2]
    chunker.add_to_buffer(chunk)
    
    assert len(chunker.buffer) == 1
    assert chunker.buffer[0].equals(chunk)


def test_flush_buffer(sample_dataframe):
    chunker = DataFrameChunker(sample_dataframe, n_approx=2, column_name="dt")
    chunk = sample_dataframe.iloc[:2]
    chunker.add_to_buffer(chunk)
    
    flushed_buffer = chunker.flush_buffer()
    
    assert len(flushed_buffer) == 1
    assert flushed_buffer[0].equals(chunk)
    assert chunker.buffer == []


def test_chunking_process(sample_dataframe):
    chunker = DataFrameChunker(sample_dataframe, n_approx=2, column_name="dt")
    chunks = list(chunker.chunk())
    
    assert len(chunks) == 4
    assert len(chunks[0]) == 3
    assert len(chunks[1]) == 3
    assert len(chunks[2]) == 3
    assert len(chunks[3]) == 1


def test_chunking_with_empty_dataframe(empty_dataframe):
    chunker = DataFrameChunker(empty_dataframe, n_approx=5, column_name="dt")
    chunks = list(chunker.chunk())
    
    assert len(chunks) == 0


def test_chunking_with_exact_fit(sample_dataframe):
    chunker = DataFrameChunker(sample_dataframe, n_approx=len(sample_dataframe), column_name="dt")
    chunks = list(chunker.chunk())
    
    assert len(chunks) == 1
    assert chunks[0].equals(sample_dataframe)
