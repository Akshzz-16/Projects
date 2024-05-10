import sys
from source.logger import logging

def errorDetails(error, errorDetail:sys):
    _,_,exc_tb = errorDetail.exc_info()
    fileName = exc_tb.tb_frame.f_code.co_filename

    errorMessage = "Error occured in python script\n name--> [{0}]\n line number--> [{1}]\n error message--> [{2}]".format(fileName, exc_tb.tb_lineno,str(error))

    return errorMessage

class CustomException(Exception):
    def __init__(self,errorMessage, errorDetail:sys):
        super().__init__(errorMessage)
        self.errorMessage = errorDetails(errorMessage, errorDetail=errorDetail)

    def __str__(self):
        return self.errorMessage
    

if __name__ == "__main__":
    try:
        a=1/0

    except Exception as e:
        logging.info(e)
        raise CustomException(e, sys)
    