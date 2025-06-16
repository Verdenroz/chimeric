import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def openai_env():
    """Ensure OPENAI_API_KEY is set for Chimeric initialization."""
    os.environ["OPENAI_API_KEY"] = "test_key"
    yield
    del os.environ["OPENAI_API_KEY"]
