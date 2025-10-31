import socket
import numpy as np

def parse_packet(packet):
    """
    Takes in the UDP packet data and parses into correct variables
    :param packet: input of the udp packet received from xPC
    :return: parsed information from packet
    """

    def parse_any_data(packet_in, data_length_list, start=0):
        idx = start
        outputs = []
        for data_len in data_length_list:
            outputs.append(packet_in[idx:idx + data_len])
            idx += data_len
        return outputs

    data_lengths = [            # should add up to 832 (July 2022)
        4,      # eTime
        1,      # feat
        2,      # dsize
        768,    # neural_data   (96 * 8 bytes)
        20,     # fpos          (5 * 4 bytes)
        20,     # target_pos    (5 * 4 bytes)
        2,      # trial_count   (uint16 - 2 bytes)
        1,      # switch1
        1,      # switch2
        1,      # switch3
        4,      # val1          (single - 4 bytes)
        4,      # val2          (single - 4 bytes)
        2,      # msCount       (uint16 - 2 bytes)
        1,      # xpcBinSize    (uint16, but likely only 1 byte after casting)
        1       # enable
    ]
    packet_np = np.array([packet]).view(np.uint8)
    # print(packet_np)
    eTime, feat, dsize, neural_data, fpos, target_pos, trial_count, switch1, switch2, switch3, val1, val2, msCount, \
        xpcBinSize, enable = parse_any_data(packet_np, data_lengths)

    msCount = msCount.view(np.ushort)  # this is xPC's iteration counter
    fpos = fpos.view(np.single)
    target_pos = target_pos.view(np.single)
    trial_count = trial_count.view(np.ushort)
    switches = [switch1, switch2, switch3]
    vals = [val1.view(np.single), val2.view(np.single)]
    xpcBinSize = xpcBinSize[0]
    # also return everything in a dict (so it's easy to pass to the decoder)
    xpc_dict = {'eTime': eTime, 'feat': feat, 'dsize': dsize, 'neural_data': neural_data, 'fpos': fpos, \
                'msCount': msCount, 'xpcBinSize': xpcBinSize, 'enable': enable, 'trial_count': trial_count,
                'target_pos': target_pos, 'xpc_switches': switches, 'xpc_vals': vals}

    return eTime, feat, dsize, neural_data, fpos, msCount, xpcBinSize, enable, \
           target_pos, trial_count, switches, vals, xpc_dict


if __name__ == '__main__':

    UDP_IP_recieve = "0.0.0.0" #get vision ip
    UDP_PORT_recieve = 11114 #get vision port
    sock_recieve = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recieve.bind((UDP_IP_recieve, UDP_PORT_recieve))
    print("Listening on port", UDP_PORT_recieve)
    while True:
        try:
            print('Waiting to receive message')
            data = sock_recieve.recv(1500)
            print("received message:")
            eTime, feat, dsize, neural_data, fpos, msCount, xpcBinSize, enable, target_pos, trial_count, \
                xpc_switches, xpc_vals, xpc_dict = parse_packet(data)
            # print('eTime:', eTime )
            # print('msCount:', msCount)
            print('fpos:', fpos)
            # print('target_pos:', target_pos)
            # print('xpc_switches:', xpc_switches)
            # print('xpc_vals:', xpc_vals)
            # print('xpc_dict:', xpc_dict)
            # print('dsize:', dsize)
            # print('neural_data:', neural_data)
            # print(neural_data.shape)
            # print('feat:', feat)
            # print('xpcBinSize:', xpcBinSize)
            # print('enable:', enable)
            print('trial_count:', trial_count)
            print('-----------------------------------')
        except Exception as e:
            print(e)
            print("Error receiving data")
            # break