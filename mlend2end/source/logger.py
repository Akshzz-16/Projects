import logging
import os
from datetime import datetime

LOGFILE = f"{datetime.now().strftime('%m_%d_%Y')}.log"
logPath = os.path.join(os.getcwd(), "logs", LOGFILE)
os.makedirs(logPath, exist_ok=True) #even if there is file keep appending folders

LOGFILEPATH = os.path.join(logPath, LOGFILE)

#override the levels of basic logging function
logging.basicConfig(
    filename=LOGFILEPATH,
    format='%(asctime)s- %(levelname)s- %(message)s - %(name)s',
    level= logging.INFO,
)

if __name__ == "__main__":
    logging.info('LOGGING HAS STARTED')

