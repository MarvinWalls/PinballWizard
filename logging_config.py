import logging

def setup_logging(log_file='gameplay.log'):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info('Logging setup complete')

