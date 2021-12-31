import inspect
import json
import os
import platform
import shutil


HOME_FOLDER = os.getcwd()
DATE_TIME_STRING = "%Y-%m-%dT%H:%M:%S.%f"


class ShapeletsConfig:
    """
    Container for configuration, loaded from JSON
    """
    test_folder = None
    # data preparation
    step = 500
    window_length = 1000
    step4negative = 5
    min_negative_last_chunk_size = 100

    # GENDIS parameters
    population_size = 5
    iterations = 10
    verbose = True
    n_jobs = 1
    mutation_prob = 0.3
    crossover_prob = 0.3
    wait = 5
    normed = True

    LOGGING_LEVEL = 20
    # CRITICAL = 50
    # ERROR = 40
    # WARNING = 30
    # INFO = 20
    # DEBUG = 10
    # NOTSET = 0

    def __init__(self, test_folder: str):
        """
        Load the configuration from json or use the defaults
        :param test_folder: config folder location.
        """
        self.json_file = test_folder + os.path.sep + "shape_conf.json"
        self.test_folder = test_folder
        if os.path.exists(self.json_file):
            try:
                self.reload()
            except ValueError:
                pass
        else:
            directory = os.path.dirname(self.json_file)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(self.json_file, "w") as fp:
                json.dump(self.json(), fp, indent=True)
            os.chmod(self.json_file, 0o644)

    def json(self):
        attributes = inspect.getmembers(self, lambda m: not (inspect.isroutine(m)))
        ret = {}
        for attr in attributes:
            if not (attr[0].startswith('__')) and attr[1] is not None:
                ret[attr[0]] = attr[1]
        # ret = {attr[0]: attr[1] for attr in attributes if not (attr[0].startswith('__')) and attr[1] is not None}
        return ret

    def update(self, conf=None):
        if conf is not None:
            for (k, v) in list(conf.items()):
                setattr(self, k, v)

        with open(self.json_file, "w") as fp:
            json.dump(self.json(), fp, indent=True)
        os.chmod(self.json_file, 0o644)

        self.reload()

    def reload(self):
        conf = json.load(open(self.json_file, "r"))
        self.__dict__.update(conf)
