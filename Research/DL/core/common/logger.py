import logging


formatter = r'%(asctime)s -- %(filename)s[line:%(lineno)d] %(levelname)s\t%(message)s'
logging.basicConfig(level=logging.INFO, formatter=formatter)

