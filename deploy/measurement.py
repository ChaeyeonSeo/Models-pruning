#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psutil
import telnetlib as tel
import sysfs_paths as sysfs
import time
import argparse

SAMPLING_RATE = 1  # 1 second delay


def get_telnet_power(telnet_connection, last_power):
    """
    Read power values using telnet.
    """
    # Get the latest data available from the telnet connection without blocking
    tel_dat = str(telnet_connection.read_very_eager())
    # print('telnet reading:', tel_dat)
    # Find latest power measurement in the data
    idx = tel_dat.rfind('\n')
    idx2 = tel_dat[:idx].rfind('\n')
    idx2 = idx2 if idx2 != -1 else 0
    idx2 = idx2 if idx2 != -1 else 0
    ln = tel_dat[idx2:idx].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power


def get_cpu_load():
    """
    Returns the cpu load as a value from the interval [0.0, 1.0]
    """
    loads = [x / 100 for x in psutil.cpu_percent(interval=SAMPLING_RATE, percpu=True)]
    if len(loads) < 8:
        loads += (8 - len(loads)) * [-1]
    return loads


def get_temps(device='rpi'):
    """
    Obtain the temp values from sysfs_paths.py
    """
    templ = []
    if device == 'rpi':
        import gpiozero
        templ.append(gpiozero.CPUTemperature().temperature)
        templ += [0, 0, 0]
    elif device == 'mc1':
        # Get temp from temp zones 0-3 (the big cores)
        for i in range(4):
            temp = float(open(sysfs.fn_thermal_sensor.format(i), 'r').readline().strip()) / 1000
            templ.append(temp)
        # Note: on the Exynos5422, cpu temperatures 5 and 7 (big cores 1 and 3, counting from 0) appear to be swapped.
        # Therefore, swap them back.
        t1 = templ[1]
        templ[1] = templ[3]
        templ[3] = t1
    return templ


def get_core_freq(core_num, device='rpi'):
    if device != 'mc1':
        return -1
    with open(sysfs.fn_cpu_freq_read.format(core_num), 'r') as f:
        return int(f.read().strip())


if __name__ == '__main__':
    # Create a text file to log the results
    parser = argparse.ArgumentParser(description='Measurement tracker for embedded systems')
    parser.add_argument('--file', type=str, default='log', help='.csv will be appended to end')
    parser.add_argument('--device', type=str, default='rpi', help='mc1 or rpi')
    args = parser.parse_args()
    device = args.device
    out_fname = 'measurements_' + device + '_' + args.file + '.csv'
    header = "time,W"
    with open(out_fname, 'w') as out_file:
        out_file.write(header)
        out_file.write("\n")

    # Measurement
    telnet_connection = tel.Telnet("192.168.4.1")
    total_power = 0.0
    for i in range(10000):
        last_time = time.time()  # time_stamp
        # System power
        total_power = get_telnet_power(telnet_connection, total_power)
        # print('Telnet power [W]:', total_power)

        # CPU load
        # usages = get_cpu_load()
        # print('CPU usage:', usages)

        # Temp for big cores
        # temps = get_temps(device)
        # print('Temperature of big cores:', temps)

        # Big cluster core frequencies
        # freq_4 = get_core_freq(core_num=4, device=device)
        # print('core4 frequency : ', freq_4)

        time_stamp = last_time
        # Data write out:
        # fmt_str = "{}," * 14
        # out_ln = fmt_str.format(time_stamp, total_power)
        with open(out_fname, 'a+') as out_file:
            out_file.write(f"{int(time_stamp)},{total_power}\n")
        elapsed = time.time() - last_time
        # We make sure we wait exactly 0.2 seconds = elapse time executing code in this for loop + the time left until
        # we reach 0.2 seconds time elapsed
        time.sleep(max(0., SAMPLING_RATE - elapsed))
