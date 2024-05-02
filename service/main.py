#!/usr/bin/python3

from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport.TSocket import TServerSocket, TSocket
from thrift.transport.TTransport import TBufferedTransport, TTransportException
from thrift.server import TServer
from threading import Thread, Condition
import argparse
from generated.chan_estimator_api import ChanEstimatorService
from generated.chan_estimator_api.ttypes import Status, Code
from enum import Enum
from queue import Queue, Empty as QueueEmpty

import time


class ServerModel(Enum):
    SINGLE_THREAD = 1
    MULTI_THREAD = 2


class EstRequest:
    def __init__(self, eMu, eMuX, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2, maintenance):
        self.eMu = eMu
        self.eMuX = eMuX
        self.eMuEma = eMuEma
        self.eNu1 = eNu1
        self.eNu2 = eNu2
        self.qMu = qMu
        self.qNu1 = qNu1
        self.qNu2 = qNu2
        self.maintenance = maintenance


class ModelStub:
    def __init__(self):
        self.is_active = False
        self.is_ready = False
        self.thread = None
        self.req_queue = Queue()
        self.resp_queue = Queue()

    def retrieve_est(self, eMu, eMuX, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2, maintenance):
        self.req_queue.put(EstRequest(eMu, eMuX, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2, maintenance))
        resp = None
        max_tries = 5
        tries = 0
        while True:
            try:
                resp = self.resp_queue.get(block=True, timeout=1)
            except QueueEmpty:
                tries += 1
                if tries > max_tries:
                    return Status(Code.INTERNAL_ERROR, 0.)
                continue
        return resp

    def start(self):
        self.is_active = True
        self.thread = Thread(target=self.loop)
        self.thread.run()

    def stop(self):
        if not self.is_active:
            raise RuntimeError("E: model thread is stopped already")
        self.is_active = False
        self.thread.join()

    def ready(self):
        return self.is_ready

    def fine_tuning(self):
        self.is_ready = False
        # stub here
        time.sleep(10)
        self.is_ready = True

    def loop(self):
        self.is_ready = True
        iter = 0
        while self.is_active:
            fine_tuning_required = iter % 10
            req = None
            try:
                req = self.req_queue.get(True, 1)
            except QueueEmpty:
                continue

            self.resp_queue.put(Status(res=Code.OK, est=req.eMuEma))
            if fine_tuning_required:
                self.fine_tuning()
            iter += 1


class ChanEstimatorHandler:
    def __init__(self, est_model):
        self.est_model = est_model

    def retrieveEst(self, eMu, eMuX, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2, maintenance):
        # TODO remove stub, add real implementation
        res = None
        if self.est_model.ready():
            res = self.est_model.retrieve_est(self, eMu, eMuX, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2, maintenance)
        else:
            res = Status(Code.TRY_LATER, 0.)
        return res


def main():
    parser = argparse.ArgumentParser(description='ML-based channel estimator')
    parser.add_argument("--sock",
                        required=True,
                        help='chan_estimator UNIX-domain socket path')
    parser.add_argument("--model",
                        required=True,
                        help='chan_estimator thread model')

    args = parser.parse_args()
    server_model = None
    if args.model == "simple":
        server_model = ServerModel.SINGLE_THREAD
    elif args.model == "multi":
        server_model = ServerModel.MULTI_THREAD
    else:
        raise RuntimeError("E: invalid server model obtained")

    est_model = ModelStub()
    est_model.start()

    handler = ChanEstimatorHandler(est_model)
    processor = ChanEstimatorService.Processor(handler)
    transport = TServerSocket(unix_socket=args.sock)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = None
    if server_model == ServerModel.SINGLE_THREAD:
        server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    elif server_model == ServerModel.MULTI_THREAD:
        # for each new connection -> new thread created
        server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)

    thrift_server_thread = Thread(target=server.serve, name='Thrift server')
    thrift_server_thread.start()


if __name__ == '__main__':
    main()
