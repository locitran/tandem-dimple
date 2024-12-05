import logging
import sys

datefmt = '%Y-%m-%d %H:%M'
log_format = '%(asctime)s - %(message)s'
formatter = logging.Formatter(log_format, datefmt=datefmt)

class Logger:
    def __init__(self, output_dir=None):
        logging.basicConfig(
            stream=sys.stdout, # to print on console
            level=logging.ERROR, # to log only error and above
            format=log_format,
            datefmt=datefmt
        )
            
        self.logger = logging.getLogger() # get root logger
        self.logger.setLevel(logging.INFO)
        self.output_dir = output_dir if output_dir else '.'

        open(f'{self.output_dir}/error.log', 'w').close()
        open(f'{self.output_dir}/info.log', 'w').close()
        open(f'{self.output_dir}/warning.log', 'w').close()

        # create console handler and set level to debug
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        # log handler for error --> log only error and above
        error_handler = logging.FileHandler(f'{self.output_dir}/error.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(MyFilter(logging.ERROR))

        # log handler for info --> log only info and above
        info_handler = logging.FileHandler(f'{self.output_dir}/info.log')
        info_handler.setLevel(logging.INFO)
        info_handler.addFilter(MyFilter(logging.INFO))

        # log handler for warning --> log only warning and above
        warning_handler = logging.FileHandler(f'{self.output_dir}/warning.log')
        warning_handler.setLevel(logging.WARNING)
        warning_handler.addFilter(MyFilter(logging.WARNING))

        # add handlers to logger
        for handler in [console_handler, error_handler, info_handler, warning_handler]:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


        # Disable propagation to the root logger
        self.logger.propagate = False


    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)

    def critical(self, message):
        self.logger.critical(message)

    def exception(self, message):
        self.logger.exception(message)

    def log(self, level, message):
        self.logger.log(level, message)

    def set_level(self, level):
        self.logger.setLevel(level)
        
class MyFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        """
        Determine if the specified record is to be logged.

        If MyFilter.level is INFO, this would allow INFO, WARNING, ERROR and CRITICAL messages to be logged.
        
        """
        return logRecord.levelno <= self.__level # log only if level is less than or equal to the level set
    