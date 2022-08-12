import logging
import os.path
import time
from colorama import Fore, Style
import sys


class Logger(object):
    def __init__(self, level=logging.INFO):
        """
        指定保存日志的文件路径，日志级别，以及调用文件
        将日志存入到指定的文件中
        :param logger:  定义对应的程序模块名name，默认为root
        """
        # 设置基础配置
        formatter = r'%(asctime)s -- %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
        logging.basicConfig(level=level,  # 指定最低的日志级别 critical > error > warning > info > debug
                    format=formatter,
                    datefmt='%a %d %b %Y %H:%M:%S'
                    )
        # 创建一个logger



        # 创建一个handler，用于写入日志文件
        # rq = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        # log_path = os.getcwd() + "/logs/"
        # log_name = log_path + rq + ".log"
        #  这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志，解决重复打印的问题
        # if not self.logger.handlers:
        #     # 创建一个handler，用于输出到控制台
        #     ch = logging.StreamHandler(sys.stdout)
        #     # ch.setLevel(logging.DEBUG)
        #
        #     # 定义handler的输出格式
        #     ch.setFormatter(formatter)
        #
        #     # 给logger添加handler
        #     self.logger.addHandler(ch)

    def logger(self, logger_name):
        return logging.getLogger(logger_name)

    # def debug(self, msg):
    #     """
    #     定义输出的颜色debug--white，info--green，warning/error/critical--red
    #     :param msg: 输出的log文字
    #     :return:
    #     """
    #     self.logger.debug(Fore.WHITE + "DEBUG - " + str(msg) + Style.RESET_ALL)
    #
    # def info(self, msg):
    #     self.logger.info(Fore.GREEN + "INFO - " + str(msg) + Style.RESET_ALL)
    #
    # def warning(self, msg):
    #     self.logger.warning(Fore.RED + "WARNING - " + str(msg) + Style.RESET_ALL)
    #
    # def error(self, msg):
    #     self.logger.error(Fore.RED + "ERROR - " + str(msg) + Style.RESET_ALL)
    #
    # def critical(self, msg):
    #     self.logger.critical(Fore.RED + "CRITICAL - " + str(msg) + Style.RESET_ALL)
