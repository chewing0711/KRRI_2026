import can

import can_parser
import decoding_functions

bus = can.Bus(
    interface="socketcan",
    channel="can0"
)

print("Waiting for CAN.")

start_time = None

try:
    while True:
        msg = bus.recv()

        if msg is None:
            continue

        recived_time, can_id, data = can_parser.parser(msg)

        if start_time is None :
            start_time = recived_time

        if can_id is not None:
            relative_time = round(recived_time - start_time, 5)
            results_1s = can_parser.collect_data(relative_time, can_id, data)

            if results_1s is not None:
                dataset = decoding_functions.decoding(results_1s)
                breakpoint()

except KeyboardInterrupt:
    print("Exit.")

finally:
    bus.shutdown()