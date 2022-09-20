#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 23:41:51 2018

@author: hwauni
"""

from multiprocessing import Process, Lock, Queue


"""
 ******************************************
 * Class Area
 ******************************************
"""
class ProcessMgr():
    def __init__(self, worker):
        self.process_list = []
        if not worker:
            return False
        self.worker = worker

    def produce_process(self, *args):
        process = Process(target=self.worker, args=(args))
        self.process_list.append(process)

    def start(self):
        for pro in self.process_list:
            pro.start()

    def stop(self):
        for pro in self.process_list:
            pro.join()