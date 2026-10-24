"""Test runs never ask GitHub for a newer release: the server start would do it in a thread (server/update_check.py).
A test of the check itself passes its own environment."""
import os

os.environ.setdefault("CRUCIBLE_NO_UPDATE_CHECK", "1")
