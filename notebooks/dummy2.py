import  signal

def signal_handler(signal, frame):
    global interrupted
    interrupted = True

class Crawler():
    def __init__(self):
        pass

    def crawl(self):
        global interrupted
        interrupted = False
        signal.signal(signal.SIGINT, signal_handler)
        self.i=0
        while True:
            for j in range(self.i,self.i+100):
                print(j)
            print("For loop ended")
            self.i+=1

            if interrupted:
                print("Exiting..")
                break

signal.signal(signal.SIGINT, signal_handler)
Crawler().crawl()