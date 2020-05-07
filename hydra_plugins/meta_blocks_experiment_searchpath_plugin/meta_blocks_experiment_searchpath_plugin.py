from hydra._internal.config_search_path import ConfigSearchPath
from hydra.plugins import SearchPathPlugin


class MetaBlocksExperimentSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath):
        search_path.append("meta-blocks", "pkg://meta_blocks.experiment.conf")
