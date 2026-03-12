import datetime
import logging
import os
import os.path as osp
from glob import glob
import termcolor


if os.name == "nt":  # Windows
    import colorama
    colorama.init()


SUCCESS_LEVEL   = 25   # 綠：Plan A 成功
EXPENSIVE_LEVEL = 35   # 粉：切片送 Grok
logging.addLevelName(SUCCESS_LEVEL,   'SUCCESS')
logging.addLevelName(EXPENSIVE_LEVEL, 'EXPENSIV')  # 8字對齊

COLORS = {
    "DEBUG":    "white",
    "INFO":     "white",
    "SUCCESS":  "green",
    "WARNING":  "yellow",
    "EXPENSIV": "magenta",
    "CRITICAL": "red",
    "ERROR":    "red",
}

ATTRS = {
    "DEBUG":    ["dark"],
    "INFO":     [],
    "SUCCESS":  ["bold"],
    "WARNING":  ["bold"],
    "EXPENSIV": ["bold"],
    "CRITICAL": ["bold"],
    "ERROR":    ["bold"],
}

# 這些 module 的 DEBUG 訊息直接靜音（避免 config_proj 存檔 spam）
SUPPRESSED_DEBUG_MODULES = {'config_proj'}


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, use_color=True):
        logging.Formatter.__init__(self, fmt)
        self.use_color = use_color

    def format(self, record):
        # 靜音指定 module 的 DEBUG
        if record.levelno == logging.DEBUG and record.module in SUPPRESSED_DEBUG_MODULES:
            record.levelname2 = ''
            record.message2 = ''
            record.asctime2 = ''
            record.module2 = ''
            record.funcName2 = ''
            record.lineno2 = ''
            return ''

        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            color = COLORS[levelname]
            attrs = ATTRS.get(levelname, [])

            def colored(text):
                return termcolor.colored(text, color=color, attrs=attrs)

            record.levelname2 = colored("{:<8}".format(levelname))
            record.message2   = colored(record.getMessage())

            asctime2 = datetime.datetime.fromtimestamp(record.created)
            record.asctime2   = termcolor.colored(asctime2, color="green")
            record.module2    = termcolor.colored(record.module,   color="cyan")
            record.funcName2  = termcolor.colored(record.funcName, color="cyan")
            record.lineno2    = termcolor.colored(record.lineno,   color="cyan")
        return logging.Formatter.format(self, record)


FORMAT = "[%(levelname2)s] %(module2)s:%(funcName2)s:%(lineno2)s - %(message2)s"


class NoEmptyFilter(logging.Filter):
    """過濾掉 formatter 回傳空字串的 record"""
    def filter(self, record):
        return True  # 讓 formatter 決定，StreamHandler 會跳過空字串


class ColoredLogger(logging.Logger):

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.INFO)
        color_formatter = ColoredFormatter(FORMAT)
        console = logging.StreamHandler()
        console.setFormatter(color_formatter)
        # 讓 StreamHandler 不輸出空字串
        original_emit = console.emit
        def filtered_emit(record):
            msg = console.format(record)
            if msg.strip():
                original_emit(record)
        console.emit = filtered_emit
        self.addHandler(console)

    def success(self, msg, *args, **kwargs):
        """綠色，Plan A 成功"""
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log(SUCCESS_LEVEL, msg, args, **kwargs)

    def expensive(self, msg, *args, **kwargs):
        """粉色，切片送 Grok"""
        if self.isEnabledFor(EXPENSIVE_LEVEL):
            self._log(EXPENSIVE_LEVEL, msg, args, **kwargs)


def setup_logging(logfile_dir: str, max_num_logs=14):
    if not osp.exists(logfile_dir):
        os.makedirs(logfile_dir)
    else:
        old_logs = glob(osp.join(logfile_dir, '*.log'))
        old_logs.sort()
        n_log = len(old_logs)
        if n_log >= max_num_logs:
            to_remove = n_log - max_num_logs + 1
            try:
                for ii in range(to_remove):
                    os.remove(old_logs[ii])
            except Exception as e:
                logger.error(e)

    logfilename = datetime.datetime.now().strftime('_%Y_%m_%d-%H_%M_%S.log')
    logfilep = osp.join(logfile_dir, logfilename)
    fh = logging.FileHandler(logfilep, mode='w', encoding='utf-8')
    fh.setFormatter(
        logging.Formatter(
            "[%(levelname)s] %(module)s:%(funcName)s:%(lineno)s - %(message)s"
        )
    )
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('BallonTranslator')
logger.setLevel(logging.DEBUG)
logger.propagate = False