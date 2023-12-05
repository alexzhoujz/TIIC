from QRec import QRec
from util.config import ModelConf

config = ModelConf("/home/xxx/algor_name.conf")
rec = QRec(config)
rec.execute()