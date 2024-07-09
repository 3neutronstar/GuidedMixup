import logging
import os
import sys
import torch.distributed as dist
import torch

def set_logging_defaults(logdir,file_name):
    # set basic configuration for logging
    # logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
    logging.basicConfig(format="[%(asctime)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir,file_name, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)],
                        datefmt = '%m-%d %I:%M:%S')

    # log cmdline argumetns
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs

    
def create_distributed_logging(logdir,log_file=None, log_level=logging.INFO, file_mode='a'):
    """Initialize and get a logger.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger()

    handlers = []
    #stream_handler = logging.StreamHandler()
    #handlers.append(stream_handler)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(os.path.join(logdir,log_file,'log.txt'), file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter('[%(asctime)s] %(message)s', "%m-%d %h:%m:%s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    return logger

def reduce_value(value, average=True):
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size < 2:  # single gpu
            return value

        with torch.no_grad():
            dist.all_reduce(value)  # sum
            if average:
                value /= world_size  # mean
    return value