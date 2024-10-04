#########################################################
# Import libs
#########################################################
import math

import numpy as np
import wlmData
import wlmConst
import ctypes
import sys
import WLM_methods
# import pyvisa
import time
import threading
# from ThorlabsPM100 import ThorlabsPM100
from datetime import date, datetime
from matplotlib import pyplot as plt

#########################################################
# You can launch a simulator with ideal signal to check resonator
#########################################################
sim = False
if sim:
    from simresonator import *
    import simresonator as TimeTagger

if not sim:
    import TimeTagger
    try:
        from MDT_COMMAND_LIB import *
    except OSError as ex:
        print("Warning:", ex)

    # Заменить путь библиотеки (скопировать из другой программы)
    mdtLib = cdll.LoadLibrary(r"C:\Users\Photon\PycharmProjects\Time_tagger\MDT_COMMAND_LIB_x64.dll")

#########################################################
# Disable Python warnings
#########################################################
import warnings
warnings.simplefilter("ignore", UserWarning)

#########################################################
# Set the global constants
#########################################################
COUNTER_BIN = 20e9
PAUSE = COUNTER_BIN*1e-12 + 1e-3
STABILIZATION_TIME = 300
REFERENCE_MEMORY_FREQUENCY = 580.035
NOISE_LEVEL_IN_COUNTS = 300
down_reference = 580.038582
upper_reference = 580.035582
plot_stab = False
plot_stab_arr_x = []
plot_stab_arr_y = []
plot_stab_arr_y2 = []
plot_stab_arr_y3 = []
plot_stab_first = True
plot_t = 0
#########################################################
# Set the DLL_PATH variable according to your environment
#########################################################
DLL_PATH = "wlmData.dll"

#########################################################
# Load DLL from DLL_PATH
#########################################################
try:
    wlmData.LoadDLL(DLL_PATH)
except:
    sys.exit("Error: Couldn't find DLL on path %s. Please check the DLL_PATH variable!" % DLL_PATH)

#########################################################
# Define specific functions
#########################################################
def CommonFunc(serialNumber):
    hdl = mdtOpen(serialNumber, 115200, 3)
    # or check by "mdtIsOpen(devs[0])"
    if (hdl < 0):
        print("Connect ", serialNumber, "fail")
        return -1
    else:
        print("Connect ", serialNumber, "successful")

    result = mdtIsOpen(serialNumber)
    print("mdtIsOpen ", result)

    id = []
    result = mdtGetId(hdl, id)
    if (result < 0):
        print("mdtGetId fail ", result)
    else:
        print(id)

    limitVoltage = [0]
    result = mdtGetLimtVoltage(hdl, limitVoltage)
    if (result < 0):
        print("mdtGetLimtVoltage fail ", result)
    else:
        print("mdtGetLimtVoltage ", limitVoltage)
    return hdl

###########################################################
def Check_X_AXiS(hdl):
    '''
    # Проверка подключения пьезовинтов
    :param hdl:
    :return:
    '''
    voltage = [0]
    result = mdtGetXAxisVoltage(hdl, voltage)
    if (result < 0):
        print("mdtGetXAxisVoltage fail ", result)
    else:
        print("mdtGetXAxisVoltage ", voltage)
    a = 0
    result = mdtSetXAxisVoltage(hdl, a)
    if (result < 0):
        print("mdtSetXAxisVoltage fail ", result)
    else:
        print("mdtSetXAxisVoltage ", 0)


def Check_Y_AXiS(hdl):
    '''
    # Проверка подключения пьезовинтов
    :param hdl:
    :return:
    '''
    voltage = [0]
    result = mdtGetYAxisVoltage(hdl, voltage)
    if (result < 0):
        print("mdtGetYAxisVoltage fail ", result)
    else:
        print("mdtGetYAxisVoltage ", voltage)

    result = mdtSetYAxisVoltage(hdl, 0)
    if (result < 0):
        print("mdtSetYAxisVoltage fail ", result)
    else:
        print("mdtSetYAxisVoltage ", 0)


def Check_Z_AXiS(hdl):
    '''
    # Проверка подключения пьезовинтов
    :param hdl:
    :return:
    '''
    voltage = [0]
    result = mdtGetZAxisVoltage(hdl, voltage)
    if (result < 0):
        print("mdtGetZAxisVoltage fail ", result)
    else:
        print("mdtGetZAxisVoltage ", voltage)

    result = mdtSetZAxisVoltage(hdl, 0)
    if (result < 0):
        print("mdtSetZAxisVoltage fail ", result)
    else:
        print("mdtSetZAxisVoltage ", 0)


#############################################################################################
def sub_stabilisation(flag, TT, ref, volt):
    if flag:
        if TT > ref:
            return volt - 0.001
        elif TT < ref:
            return volt + 0.001
        elif TT == ref:
            return volt
    else:
        if TT > ref:
            return volt + 0.001
        elif TT < ref:
            return volt - 0.001
        elif TT == ref:
            return volt


def Nail_based_stabilisation_improved_v1(hdl, file, counts_max, counter, count_reference, volt_reference, low_perc):
    volt = volt_reference  # Напряжение, где наблюдается референсный счёт
    abortion_flag = False
    measurement_location_flag = True # to the right of peak
    low_count = False # low counts detection flag
    dots = []
    amount = 0
    max_allowed_amount = 3 # max value to measure when lower than low_perc * counts_max
    d = count_reference  # Величина счета, на которую стабилизируемся
    while True and (volt > 0) and (volt < 100):
        if (volt < 0) or (volt > 100):
            volt = volt_reference
        a = counter.getData()
        TT = a.flatten()[0]
#        time.sleep(2e-3)
        print('Reference counts: ', d)
        print(f"Current counts: {TT}\n")

        mdtSetXYZAxisVoltage(hdl, volt, volt, volt)

        if TT < counts_max * low_perc and measurement_location_flag:
            dots.append(TT)
            volt += 0.001
            amount += 1
            low_count = True
        elif TT < counts_max * low_perc and (not measurement_location_flag):
            dots.append(TT)
            volt -= 0.001
            amount += 1
            low_count = True
        elif TT > counts_max * low_perc:
            if low_count:
                low_count = False
                amount = 0
                dots.clear()
            volt = sub_stabilisation(measurement_location_flag, TT, d, volt)
        if low_count and (amount > max_allowed_amount):
            dots_deriv = np.gradient(dots, 0.001)
            if np.mean(dots_deriv) <= 0:
                if measurement_location_flag:
                    measurement_location_flag = False
                else:
                    measurement_location_flag = True
            low_count = False
            amount = 0
            dots.clear()
        if TT < counts_max // 6:
            abortion_flag = True
            break
    return abortion_flag


def plot_stab_plot():
    global plot_stab_first
    if plot_stab_first:
        plot_stab_first = False
        plt.clf()
    plt.plot(plot_stab_arr_x,plot_stab_arr_y,'b-')
    plt.plot(plot_stab_arr_x,plot_stab_arr_y2,'g-')
    plt.plot(plot_stab_arr_x,plot_stab_arr_y3,'r-')
    plt.draw()

def Nail_based_stabilisation(hdl, file, counts_max, counter, count_reference, volt_reference, low_perc,
                             START_TIME, TIME_LIMIT):
    global plot_t
    chan = 2 # channel to measure laser frequency
    call_improved_function = False
    abortion_flag = False
    data_list = []
    frequency_list = []
    data_list_max_size = 1000
    reference = wlmData.dll.ConvertUnit(REFERENCE_MEMORY_FREQUENCY, wlmConst.cReturnWavelengthVac,
                                        wlmConst.cReturnFrequency)
    index = 0
    flag = False
    if call_improved_function:
        print("HEREEEE")
        abortion_flag = Nail_based_stabilisation_improved_v1(hdl, file, counts_max, counter,
                                                             count_reference, volt_reference, low_perc)
        print("HEREEEE")
        return abortion_flag
    else:
        volt = volt_reference # Напряжение, где наблюдается референсный счёт

        dt = 2e9/1e12
        PID_P = 20.5/100/counts_max*2 # из симулятора    PID_I = 5e-7
        PID_SUM = 0
        stab_direction = 1
        prev_TT = 0
        dv = 0.03
        counter.start()

        while True and (volt > 0) and (volt < 100):

            mdtSetXYZAxisVoltage(hdl, volt, volt, volt)

            # counter = TimeTagger.Counter(tagger=tagger, channels=[1], binwidth=int(1e10), n_values=1)
            d = count_reference # Величина счета, на которую стабилизируемся
            #counter.startFor(2e9)
            #counter.waitUntilFinished()
            #time.sleep(21e-3)
            time.sleep(PAUSE)
            a = counter.getData()
            TT = a.flatten()[0]
            #time.sleep(2e-3)
            #print('Reference counts: ', d)
            #print(f"Current counts: {TT}\n")
            '''
            if TT<prev_TT:
                stab_direction = -stab_direction
            volt += stab_direction*0.04
            prev_TT = TT
            '''

            # Стабилизация

            if TT > d:
                volt -= dv
            elif TT < d:
                volt += dv

            if index < data_list_max_size:
                data_list.append(TT)
                frequency_list.append(math.fabs(wlmData.dll.GetFrequencyNum(chan, 0) - reference))
                index += 1
            else:
                arr = np.array(data_list)
                arr2 = np.array(frequency_list)
                mean = arr.mean()
                mean2 = arr2.mean()
                std = arr.std()
                std2 = arr2.std()
                file.write(f"Stabilisation. {data_list_max_size} points statistics. Mean_count: {round(mean, 3)}, Deviation_count: {round(std, 3)}\n")
                file.write(f"Stabilisation. {data_list_max_size} points statistics. Mean absolute deviation: {round(mean2, 10)} [THz]\n")
                file.flush()
                index = 0
                data_list.clear()
                frequency_list.clear()

            #elif TT == d:
            #    volt = volt

            '''
            err = TT-d
            dv = err*PID_P
            PID_SUM += err
            volt -= dv
            volt -= PID_SUM*PID_I

            if plot_stab:
                plot_t+=1
                plot_stab_arr_x.append(plot_t)
                #plot_stab_arr_y.append(volt)
                plot_stab_arr_y.append(err)
                plot_stab_arr_y2.append(dv*counts_max**2/(20.5/100))
                plot_stab_arr_y3.append(PID_SUM*PID_I*counts_max**2/(20.5/100))
                plot_stab_plot()
            '''
            if time.time() - START_TIME >= TIME_LIMIT:
                # Может выдать 0 значения, если время выйдет сразу после выдачи статистики за предыдущие data_list_max_size точки
                arr = np.array(data_list)
                arr2 = np.array(frequency_list)
                mean = arr.mean()
                mean2 = arr2.mean()
                std = arr.std()
                std2 = arr2.std()
                file.write(f"Stabilisation. {data_list_max_size} points statistics. Mean_count: {round(mean, 3)}, Deviation_count: {round(std, 3)}\n")
                file.write(f"Stabilisation. {data_list_max_size} points statistics. Mean_frequency: {round(mean2, 8)} [THz], Deviation_frequency: {round(std2, 8)} [THz]\n")
                file.flush()
                data_list.clear()
                frequency_list.clear()
                raise Exception(f"Time limit: {TIME_LIMIT} s. exceeded")
            if TT < 0.2*counts_max:
                arr = np.array(data_list)
                arr2 = np.array(frequency_list)
                mean = arr.mean()
                mean2 = arr2.mean()
                std = arr.std()
                std2 = arr2.std()
                file.write(f"Stabilisation. {data_list_max_size} points statistics. Mean_count: {round(mean, 3)}, Deviation_count: {round(std, 3)}\n")
                file.write(f"Stabilisation. {data_list_max_size} points statistics. Mean_frequency: {round(mean2, 8)} [THz], Deviation_frequency: {round(std2, 8)} [THz]\n")
                file.flush()
                data_list.clear()
                frequency_list.clear()
                abortion_flag = True
                break
        return abortion_flag

def scan(counter, hdl, dots=500, start=25., stop=75., step=0.1, make_scanned_plot_flag=True):
    # initialise some vars
    index = 0
    counts_max = 0
    volt_max = 0
    mod_width = 0
    N = dots
    volt = start
    print('Start voltage: ', volt)

    if stop-start < dots*step:
        print(f"Voltage limit exceed. Step or dots should be corrected")
        return

    voltage_list = []
    mdtSetXYZAxisVoltage(hdl, volt, volt, volt)  # setting start distance in the mirror with piezo

    counter_data = []
    for _ in range(N):
        volt += step
        voltage_list.append(round(volt, 3))
        mdtSetXYZAxisVoltage(hdl, volt, volt, volt)
        counter.startFor(COUNTER_BIN)
        counter.waitUntilFinished()
        #time.sleep(4e-3)
        counter_data.append(counter.getData())

    #counter_data = counter.getData()
    counter_data = np.array(counter_data)
    counts = counter_data.flatten().tolist().copy()

    # Finding max
    index = counts.index(max(counts))
    counts_max = counts[index]
    volt_max = voltage_list[index]

    # Finding width
    mod_width = 1.5
    # mod_width = find_width_mod(voltage_list, counts, index)
    # print(f"Mod width {mod_width}")
    # print(f"Count max {counts_max}")
    # print(f"Volt max {volt_max}")
    # If we want to draw the plot of scanned area
    # Remark: you can find plots in the folder of the current project
    if make_scanned_plot_flag:
        save_plot(voltage_list, counts, "Voltage", "Counts in channel", "Resonator")
        show_plot(voltage_list, counts, "Voltage", "Counts in channel", "Resonator")

    return mod_width, index, counts_max, volt_max


def save_plot(args, func, x_lbl, y_lbl, plot_name: str):
    plt.plot(args, func, 'black', linewidth=0.5)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.grid(True)
    plt.savefig(plot_name + str(round(time.time())) + ".png")


def show_plot(args, func, x_lbl, y_lbl, plot_name: str):
    #return
    plt.ion()
    plt.plot(args, func, 'black', linewidth=0.5)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.grid(True)
    plt.title(plot_name)
    plt.show()
    plt.pause(0.0001)


def stabilisation_helper_with_restart(hdl, counter, file, stabilisation_timer_pause, N,
                                      step, volt, counts_max, perc, low_perc, START_TIME, TIME_LIMIT):
    abortion_flag = False
    now1 = datetime.now()
    counts_ref_noisy = (1. - perc) * NOISE_LEVEL_IN_COUNTS + perc * counts_max  # if noise is high this increases ref-ce
    dt_string1 = now1.strftime("%d/%m/%Y %H:%M:%S")
    file.write(f"Starting to search for the reference maximum count at {dt_string1}.\n")
    for i in range(N):
        if volt>100:
            break
        volt += step
        mdtSetXYZAxisVoltage(hdl, volt, volt, volt)
        #time.sleep(stabilisation_timer_pause)
        counter.startFor(COUNTER_BIN)
        counter.waitUntilFinished()
        counts = (counter.getData()).flatten()[0]
        print(f'Counts: {counts} Dot Number: {i}')

        if counts >= counts_ref_noisy:
            # if time.time() - START_TIME < TIME_LIMIT:
            #     break
            now1 = datetime.now()
            dt_string1 = now1.strftime("%d/%m/%Y %H:%M:%S")
            file.write(f"Stabilisation started at {dt_string1}.\nCurrent reference counts: {counts}, Current voltage: {volt}\n")

            abortion_flag = Nail_based_stabilisation(hdl, file, counts_max, counter, counts, volt, low_perc,
                                                     START_TIME, TIME_LIMIT)

            # stabilize_with_grad2(stab_counter, hdl, 3, 0.001, counts_max, volt)
            # stabilize(new_counter, hdl, counts, stabilisation_timer_pause, 0.2, counts_max, volt)
            if abortion_flag:
                now1 = datetime.now()
                dt_string1 = now1.strftime("%d/%m/%Y %H:%M:%S")
                file.write(f"Stabilisation abandoned (lost peak) at {dt_string1}.\n")
                break
            else:
                now1 = datetime.now()
                dt_string1 = now1.strftime("%d/%m/%Y %H:%M:%S")
                file.write(f"Stabilisation didn't start, however peak has bean found. Check stabilisation before cycle\n")
                if time.time() - START_TIME < TIME_LIMIT:
                    raise Exception(f"Time limit: {TIME_LIMIT} s. exceeded")
    return abortion_flag


def MDT693BExample(serialNumber, tagger, file, TIME_LIMIT, stabilisation_timer_pause=0.):

    # Check the connectivity
    hdl = CommonFunc(serialNumber)
    print('hdl:', hdl)
    if hdl < 0:
        return

    # Check piezo-elements
    Check_X_AXiS(hdl)
    Check_Y_AXiS(hdl)
    Check_Z_AXiS(hdl)

    N = 500
    step = 0.1
    volt = 0

    #counter = TimeTagger.Counter(tagger=tagger, channels=[3], binwidth=int(2e9), n_values=N)  # tagger counter
    new_counter = TimeTagger.Counter(tagger=tagger, channels=[3], binwidth=int(COUNTER_BIN), n_values=1)  # tagger counter
    parameters = scan(new_counter, hdl, dots=N, start=volt, stop=volt + N*step, step=step) # search for maximum count


    #stab_counter = TimeTagger.Counter(tagger=tagger, channels=[3], binwidth=int(2e9), n_values=3)  # tagger counter
    print('Start voltage: ', volt)
    mdtSetXYZAxisVoltage(hdl, volt, volt, volt)  # setting start distance in the mirror with piezo

    mod_width = parameters[0]
    index = parameters[1]
    counts_max = parameters[2]
    volt_max = parameters[3]

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    file.write(f"Current date: \n{dt_string}\n")
    print('Max voltage: ', volt_max)
    file.write(f"Voltage for maximum counts: {volt_max}\n")

    restarts = 0 # number of stabilisation restarts
    N = 500 # number of voltage dots
    step = 0.1 # value for each change of voltage
    #volt = 0 # start voltage
    volt = max(0., volt_max - 10)  # start voltage
    abortion_flag = False
    counts_accuracy_manage_flag = False
    fails = 0
    max_fails = 10 # max fails allowed to restart the search of "percent * counts_max" value
    percent = 50 # percentage of counts in channel for reference
    low_counts_level_percent = 50 # percentage to consider the counts low
    START_TIME = time.time()
    while True:
        if fails > max_fails: # we decrease reference counts to stabilise on if it's too high to be detected
            percent -= 5 # during the second scan operation in "stabilisation_helper_with_restart"
            fails = 0
            counts_accuracy_manage_flag = False
            now2 = datetime.now()
            dt_string2 = now2.strftime("%d/%m/%Y %H:%M:%S")
            file.write(f"Decreased max reference count at {dt_string2}. Current reference count we stabilise on: {percent*counts_max} ({percent}%)\n")
            file.flush()
        abortion_flag = stabilisation_helper_with_restart(hdl, new_counter, file, stabilisation_timer_pause, N,
                                                          step, volt, counts_max, percent / 100,
                                                          low_counts_level_percent / 100, START_TIME, TIME_LIMIT)

        volt=0
        if not abortion_flag and not counts_accuracy_manage_flag:
            counts_accuracy_manage_flag = True
        if abortion_flag:
            restarts += 1
            abortion_flag = False
            if counts_accuracy_manage_flag:
                counts_accuracy_manage_flag = False
                fails = 0
            file.write(f"Stabilisation failed. Current restarts count: {restarts}\n")
            file.flush()
        else:
            now1 = datetime.now()
            dt_string1 = now1.strftime("%d/%m/%Y %H:%M:%S")
            file.write(f"No stabilisation has occurred. Restart at {dt_string1}. Current fails count: {fails+1}\n")
            file.flush()
        if counts_accuracy_manage_flag:
            fails += 1
        if time.time() - START_TIME < TIME_LIMIT:
            raise Exception(f"Time limit: {TIME_LIMIT} s. exceeded")


def main():
    print("*** MDT device python example ***")
    WLM_available = False
    try:
        file = open("Kirill_Stabilisation_log.txt", 'a+')
        tagger = TimeTagger.createTimeTagger()
        devs = mdtListDevices()
        print(devs)
        if(len(devs)<=0):
           print('There is no devices connected')
           exit()
        if wlmData.dll.GetWLMCount(0) == 0:
            print("There is no running wlmServer instance(s).")
        else:
            # Read Type, Version, Revision and Build number
            Version_type = wlmData.dll.GetWLMVersion(0)
            Version_ver = wlmData.dll.GetWLMVersion(1)
            Version_rev = wlmData.dll.GetWLMVersion(2)
            Version_build = wlmData.dll.GetWLMVersion(3)
            print("WLM Version: [%s.%s.%s.%s]" % (Version_type, Version_ver, Version_rev, Version_build))
            WLM_available = True
        for mdt in devs:
            if(mdt[1] == "MDT693B" and WLM_available):
                # Внутри resonator_info_maker есть time_limit, который ограничивает время сканирования
                WLM_methods.resonator_info_maker(down_reference, upper_reference)
                wlmData.dll.SetDeviationMode(True)
                wlmData.dll.SetDeviationReference(REFERENCE_MEMORY_FREQUENCY)
                # Можно сделать вывод картинки в файл, чтобы не прерывать выполнение программы
                MDT693BExample(mdt[0], tagger, file, STABILIZATION_TIME)
                break

    except KeyboardInterrupt:
        print('Interrupted')
    except Exception as ex:
        print("Warning:", ex)
    finally:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        file.write(f"Closed at: \n{dt_string}\n")
        file.write("--------------------------------------------------------------------\n")
        print(f"closing file {file.name}")
        file.close()
        TimeTagger.freeTimeTagger(tagger)
        print("*** End ***")
# threading.Thread(target=main, daemon=True).start()
# input('Press <Enter> to exit.')
main()
