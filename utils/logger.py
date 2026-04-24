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


_run_log_dir: str = ''
_run_log_records: list = []
_run_log_handler: logging.Handler = None
_RUN_LOG_FMT = logging.Formatter("[%(levelname)-8s] %(module)s:%(funcName)s:%(lineno)s - %(message)s")


class _RunLogCapture(logging.Handler):
    """攔截所有 log record 存到 buffer，flush 時寫入檔案。"""
    def emit(self, record):
        _run_log_records.append(record)


def start_run_log():
    """任務開始：清空上輪 buffer，開始收集本輪 log。"""
    global _run_log_handler
    _run_log_records.clear()
    if _run_log_handler is None:
        _run_log_handler = _RunLogCapture()
        _run_log_handler.setLevel(logging.DEBUG)
        logger.addHandler(_run_log_handler)


def flush_run_log(reason: str = 'finished'):
    """任務結束/停止/關閉：把 buffer 寫到新的時間戳檔案。"""
    if not _run_log_records or not _run_log_dir:
        return
    ts = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    fname = f'run_{ts}_{reason}.log'
    fpath = osp.join(_run_log_dir, fname)
    try:
        with open(fpath, 'w', encoding='utf-8') as f:
            for record in _run_log_records:
                try:
                    f.write(_RUN_LOG_FMT.format(record) + '\n')
                except Exception:
                    pass
    except Exception as e:
        logger.error(f'flush_run_log 失敗: {e}')
    _run_log_records.clear()


def setup_logging(logfile_dir: str, max_num_logs=14):
    global _run_log_dir
    _run_log_dir = logfile_dir
    if not osp.exists(logfile_dir):
        os.makedirs(logfile_dir)
    else:
        # 保留最新的 session log（*.log）和 run log（run_*.log），各自計算上限
        old_logs = sorted(glob(osp.join(logfile_dir, '_*.log')))
        n_log = len(old_logs)
        if n_log >= max_num_logs:
            for p in old_logs[:n_log - max_num_logs + 1]:
                try: os.remove(p)
                except Exception: pass
        # run log 最多保留 50 個
        old_runs = sorted(glob(osp.join(logfile_dir, 'run_*.log')))
        n_run = len(old_runs)
        if n_run >= 50:
            for p in old_runs[:n_run - 50 + 1]:
                try: os.remove(p)
                except Exception: pass

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