from jtop import jtop, JtopException

total_pwr = 0
with jtop() as jetson:
    # monitor current power consumption for ~50s
    for n_measurements in range(1,101):
        if jetson.ok():
            cur_pwr = jetson.power[1]['CPU GPU CV']['cur']
            total_pwr = total_pwr + cur_pwr
            print(total_pwr/n_measurements)