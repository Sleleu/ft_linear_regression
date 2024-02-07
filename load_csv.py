import pandas as pd


def load(path: str) -> pd.DataFrame:
    """
    Load a CSV file from the specified path into a Pandas DataFrame.

    Parameters:
    - path (str): The path to the CSV file to be loaded.

    Returns:
    - The loaded DataFrame if successful, None if an error occurs.
    """
    HANDLED_ERRORS = (FileNotFoundError, PermissionError,
                      ValueError, IsADirectoryError)
    try:
        df = pd.read_csv(path)
        print(f"Loading dataset of dimensions {df.shape}")
        return df
    except HANDLED_ERRORS as error:
        print(f"{__name__}: {type(error).__name__}: {error}")
        return exit(1)
