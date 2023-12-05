from QRec import QRec
from util.config import ModelConf
import time

if __name__ == '__main__':
    s = time.time()
    # 超参数实验用
    # for i in range(9):
    #     name_tmp = f'./config/TIIC{i}.conf'
    #     conf = ModelConf(name_tmp)
    #     recSys = QRec(conf)
    #     recSys.execute()
    #     e = time.time()
    #     print("Running time: %f s" % (e - s))
    name_tmp = f'./config/TIIC.conf'
    conf = ModelConf(name_tmp)
    recSys = QRec(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
