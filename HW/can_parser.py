CAN_ID_LIST = ['0x371', '0x220', '0x2B0', '0x386']

def parser(msg):
    recived_time = msg.timestamp
    can_id = hex(msg.arbitration_id)
    dlc = msg.dlc
    data = msg.data.hex(' ')

    if can_id in CAN_ID_LIST :
        return recived_time, can_id, dlc, data
    else :
        return recived_time, None, None, None