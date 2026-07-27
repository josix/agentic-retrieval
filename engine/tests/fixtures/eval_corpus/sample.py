"""Sample utility module for eval fixtures."""


def parse_args(argv):
    """Parse a list of key=value strings into a dict."""
    result = {}
    for item in argv:
        key, _, value = item.partition("=")
        result[key] = value
    return result


class ConfigLoader:
    """Loads configuration from a mapping and applies defaults."""

    def __init__(self, defaults):
        self.defaults = defaults

    def load(self, overrides):
        merged = dict(self.defaults)
        merged.update(overrides)
        return merged
