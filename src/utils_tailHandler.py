import logging
import logging.config
import logging.handlers

# Configuration
# ::

audit="""
version: 1
disable_existing_loggers: False
handlers:
  console:
    class: logging.StreamHandler
    stream: ext://sys.stderr
    formatter: basic
  audit_file:
    class: logging.FileHandler
    filename: audit.log
    encoding: utf-8
    formatter: detailed
formatters:
  basic:
    style: "{"
    format: "{levelname:s}:{name:s}:{message:s}"
  detailed:
    style: "{"
    format: "{levelname:s}:{name:s}:{asctime:s}:{message:s}"
    datefmt: "%Y-%m-%d %H:%M:%S"
loggers:
  debug:
    handlers: [console]
    level: DEBUG
    propagate: False
  audit:
    handlers: [consol,audit]
    level: INFO
    propagate: False
root:
  handlers: [console]
  level: INFO
"""


tail="""
version: 1
disable_existing_loggers: False
handlers:
  console:
    class: logging.StreamHandler
    stream: ext://sys.stderr
    formatter: basic
  tail:
    (): src.utils_tailHandler.TailHandler
    target: cfg://handlers.debug_file
    capacity: 5
  debug_file:
    class: logging.FileHandler
    filename: tail_file.log
    encoding: utf-8
    formatter: detailed
formatters:
  basic:
    style: "{"
    format: "{levelname:s}:{name:s}:{message:s}"
  detailed:
    style: "{"
    format: "{levelname:s}:{name:s}:{asctime:s}:{message:s}"
    datefmt: "%Y-%m-%d %H:%M:%S"
loggers:
  tail:
    handlers: [tail]
    level: DEBUG
    propagate: False
root:
  handlers: [console]
  level: INFO
"""

#    (): Scripts.utils.TailHandler


def log_to( *names ):
    if len(names) == 0:
        names= ['logger']
    def concrete_log_to( class_ ):
        for log_name in names:
            setattr( class_, log_name, logging.getLogger(
                log_name + "." + class_.__qualname__ ) )
        return class_
    return concrete_log_to


def logged( class_ ):
    class_.logger= logging.getLogger('debug.' + class_.__qualname__)
    return class_


class TailHandler( logging.handlers.MemoryHandler ):
    def shouldFlush(self, record):
        """
        Check for buffer full or a record at the flushLevel or higher.
        """
        if record.levelno >= self.flushLevel: return True
        while len(self.buffer) >= self.capacity:
            self.acquire()
            try:
                del self.buffer[0]
            finally:
                self.release()

