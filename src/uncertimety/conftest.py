import pytest
from uncertimety.random_control import seed_all


# This ensures that every test starts with the same RNG seed unless overridden
@pytest.fixture(autouse=True)
def seed_rngs():
    seed_all(42)
