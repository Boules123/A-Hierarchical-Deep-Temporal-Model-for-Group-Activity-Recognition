import os
import sys
import logging
from datetime import datetime


class Logger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.log_dir = os.path.join(exp_dir, 'logs')
        self._create_log_directory()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'training_{timestamp}.log')
        
        self.logger = self._setup_logger()
        self.log_file_handle = self._open_log_file()
    
    def _create_log_directory(self):
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _setup_logger(self):
        logger_name = f'kaggle_logger_{datetime.now().strftime("%H%M%S")}'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _open_log_file(self):
        return open(self.log_file, 'a', encoding='utf-8')
    
    def _get_timestamp(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _direct_write(self, level, message):
        timestamp = self._get_timestamp()
        formatted_message = f"{timestamp} [{level}] {message}"
        
        self.log_file_handle.write(f"{formatted_message}\n")
        self.log_file_handle.flush()
        
        print(formatted_message)
        sys.stdout.flush()
            
    
    def info(self, message):
        self.logger.info(message)
        
    def warning(self, message):
        
        self.logger.warning(message)
        
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        original_level = self.logger.level
        if original_level > logging.DEBUG:
            self.logger.setLevel(logging.DEBUG)
        
        self.logger.debug(message)
        
        self.logger.setLevel(original_level)
        
    def close(self):
        if hasattr(self, 'log_file_handle') and self.log_file_handle:
            self.log_file_handle.close()
        
        if hasattr(self, 'logger') and self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

    def __del__(self):
        self.close()


def setup_logging(exp_dir):
    """Set up logging for the experiment."""
    return Logger(exp_dir)


def force_display(message):
    """Force immediate display in Kaggle notebooks using IPython if available."""
    from IPython.display import display, HTML
    display(HTML(f"<pre>{message}</pre>"))