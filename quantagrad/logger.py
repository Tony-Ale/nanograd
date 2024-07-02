import logging

class LoggerSetup:

    debug = False

    @classmethod
    def set_debugger(cls, debug):
        """This Serves as a top level form of setting debug mode for any class"""
        cls.debug = debug


    @staticmethod
    def setup_logger(name, level='DEBUG'):
        logger = logging.getLogger(name)
        logger.setLevel(level=level)
        handler = logging.StreamHandler()
        logger.addHandler(handler)

        return logger