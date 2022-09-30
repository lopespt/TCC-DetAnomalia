
from shutil import which

import logging as log



log.basicConfig(filename="../src/logs/log/teste.log", filemode="a", format="%(message)s")


def writeLog(text):
      log.warning(text)

# def writeLog(text):

#     with open("../src/logs/log/teste.txt", "a") as log:
#       log.writelines(text + "\n")
