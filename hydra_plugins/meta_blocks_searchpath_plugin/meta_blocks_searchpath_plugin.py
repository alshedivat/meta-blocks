from hydra._internal.config_search_path import ConfigSearchPath
from hydra.plugins import SearchPathPlugin


class MetaBlocksSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath):
        search_path.append("meta-blocks", "pkg://meta_blocks.conf")
