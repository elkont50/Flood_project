
from multie_linear import model
from datetime import datetime, timedelta
from threading import Timer

#Timer
x=datetime.today()
y=x.replace(day=x.day, hour=2, minute=0, second=0, microsecond=0)+ timedelta(days=1)
delta_t=y-x
secs=delta_t.total_seconds()


def main():
    print("Done..!")
if __name__ == '__main__':
    #Timer 
    #lambda:
    #t = Timer(secs, model())
    model()
    #t.start()
