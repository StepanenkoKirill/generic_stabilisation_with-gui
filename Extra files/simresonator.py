import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate


VOLTAGE  = 0
def gauss(x,b,sigma):
    return math.exp(-0.5*((x-b)/sigma)**2)


def train3(x,sigma):
    a = 0
    for b in range(0,5):
        t_b = 19+b*25
        #amp = gauss_train2(t_b,0,sigma2)
        #a += amp*1/(sigma*2*math.pi)*exp(-0.5*((x-t_b)/sigma)**2)

        amp = gauss(t_b,x,sigma)
        amp2 = gauss(t_b+5*sigma,x,sigma)
        #a += 100*(amp+amp2)
        a += 100*(amp)
    a+=20
    return a
sigma = 20.5/67
RESONATOR_VOLT = np.linspace(-50,150,2000)
RESONATOR_DATA = [ train3(_v,sigma) for _v in RESONATOR_VOLT]
RESONATOR_FUNC  = interpolate.interp1d(RESONATOR_VOLT, RESONATOR_DATA)
RESONATOR_DRIFT = 0
#plt.plot(RESONATOR_DATA)
#plt.show()

def mdtListDevices():
    return [["simdev","MDT693B"]]

class Counter:
    def __init__(self,tagger,channels,binwidth,n_values):
        pass
    def getData(self):
        random_step = 10*2e9/1e12*1
        r2 = np.random.normal(1,1)
        global VOLTAGE
        global RESONATOR_DRIFT
        if np.random.random()>0.5:
            RESONATOR_DRIFT += random_step*r2
        else:
            RESONATOR_DRIFT -= random_step*r2
        print(VOLTAGE, RESONATOR_DRIFT)
        #RESONATOR_DRIFT += np.random.normal(0,0.01*2e9/1e12)
        f = RESONATOR_FUNC(VOLTAGE+RESONATOR_DRIFT)
        f = np.random.poisson(f)
        return np.array([f])
    def startFor(self, dur):
        return True
    def start(self):
        return True
    def waitUntilFinished(self):
        return True

def createTimeTagger():
    return 1

def freeTimeTagger(tagger):
    return 1

def mdtOpen(ser,baud,n):
    return 1
def mdtIsOpen(ser):
    return True
def mdtGetId(hdl,_id):
    return True
def mdtGetLimtVoltage(hdl,volt):
    return True
def mdtGetXAxisVoltage(hdl,volt):
    return 0
def mdtGetYAxisVoltage(hdl,volt):
    return 0
def mdtGetZAxisVoltage(hdl,volt):
    return 0
def mdtSetXAxisVoltage(hdl,volt):
    return 0
def mdtSetYAxisVoltage(hdl,volt):
    return 0
def mdtSetZAxisVoltage(hdl,volt):
    return 0
# https://stackoverflow.com/questions/7859147/round-in-numpy-to-nearest-step
def getRoundedThresholdv1(a, MinClip):
    return round(float(a) / MinClip) * MinClip
def mdtSetXYZAxisVoltage(hdl,volt1,volt2,volt3):
    global VOLTAGE
    VOLTAGE = getRoundedThresholdv1(volt1,0.002)
    plt.pause(0.001)
    return 0
def getVoltageDrift():
    global RESONATOR_DRIFT
    return RESONATOR_DRIFT
