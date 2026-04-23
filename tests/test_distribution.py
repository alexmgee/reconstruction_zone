"""Tests for distribution flag and version consistency."""

from prep360.distribution import is_gumroad, DISTRIBUTION


def test_default_distribution_is_github():
    """The source tree must default to 'github', never 'gumroad'."""
    assert DISTRIBUTION == "github"


def test_is_gumroad_returns_false_by_default():
    assert is_gumroad() is False


def test_version_strings_match():
    """prep360 and reconstruction_gui must declare the same version."""
    import prep360
    from reconstruction_gui._version import __version__
    assert prep360.__version__ == __version__
