import logging
import logging.handlers


def CreateLogger(logger_name, filename):
    # Create Logger
    logger = logging.getLogger(logger_name)

    # Check handler exists
    if len(logger.handlers) > 0:
        return logger  # Logger already exists

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    #     formatter = logging.Formatter('\n[%(levelname)s|%(name)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    # Create Handlers
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(filename)
    fileHandler.setLevel(logging.DEBUG)
    #     streamHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    return logger
