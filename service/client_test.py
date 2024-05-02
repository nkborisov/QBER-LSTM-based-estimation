from time import sleep

from thrift.transport.TSocket import TSocket

from generated.chan_estimator_api import ChanEstimatorService
from thrift.transport.TTransport import TBufferedTransport, TTransportException
from thrift.protocol import TBinaryProtocol

THRIFT_CONNECT_RETRY_TIMEOUT = 5


def open_thrift_transport(thrift_transport):
    retries = 100
    while retries:
        try:
            thrift_transport.open()
            break
        except TTransportException:
            retries -= 1
            sleep(THRIFT_CONNECT_RETRY_TIMEOUT)
            continue
        except IOError:
            raise
    else:
        raise IOError("Could not connect to thrift server after 100 retries")


def main():
    transp = TBufferedTransport(TSocket(unix_socket="/home/galactic/chanest.sock"))
    open_thrift_transport(transp)
    client = ChanEstimatorService.Client(TBinaryProtocol.TBinaryProtocol(transp))
    ema_est = 0.008
    est = client.retrieveEst(0.01, ema_est, 0.02, 0.05, 0.003, 0.0004, 0.00005, False)
    assert est == ema_est


if __name__ == '__main__':
    main()
