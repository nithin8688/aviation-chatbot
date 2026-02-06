"""
Logging System - PHASE 4
Comprehensive logging for production observability

WHY THIS EXISTS
───────────────
Print statements are fine for development, but production needs:
• Structured logs (JSON format)
• Log levels (DEBUG, INFO, WARNING, ERROR)
• Performance metrics
• Error tracking
• Log rotation

FEATURES
────────
• Request/response logging
• Performance tracking (timing, token counts)
• Error tracking with stack traces
• Query analytics
• User behavior tracking
"""

import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps
import traceback


# ============================================================================
# LOGGER SETUP
# ============================================================================
class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add custom fields from extra kwargs
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", "funcName",
                          "levelname", "lineno", "module", "msecs", "message",
                          "pathname", "process", "processName", "relativeCreated",
                          "thread", "threadName", "exc_info", "exc_text", "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data)


def setup_logger(
    name: str = "aviation_chatbot",
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    use_json: bool = False
) -> logging.Logger:
    """
    Setup a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files (None = console only)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        use_json: Use JSON formatting (True) or plain text (False)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    if use_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        )
    
    logger.addHandler(console_handler)
    
    # File handler (if log_dir specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        file_handler = logging.FileHandler(
            log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        
        if use_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                )
            )
        
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
_logger = None

def get_logger() -> logging.Logger:
    """Get or create the global logger"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


# ============================================================================
# DECORATORS
# ============================================================================
def log_execution(operation_name: Optional[str] = None):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution("embedding")
        def compute_embedding(text):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            name = operation_name or func.__name__
            
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                
                logger.info(f"✅ {name}: {duration:.0f}ms")
                
                return result
            
            except Exception as e:
                duration = (time.time() - start) * 1000
                logger.error(f"❌ {name} failed: {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_info(message: str):
    """Quick info log"""
    get_logger().info(message)


def log_error(message: str, exc_info=False):
    """Quick error log"""
    get_logger().error(message, exc_info=exc_info)


def log_warning(message: str):
    """Quick warning log"""
    get_logger().warning(message)