import socket
import time
import pyvjoy

# Create an instance of vJoy device
client = socket.socket()

# Initialize vJoy device instance
j = pyvjoy.VJoyDevice(1)
j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
j.set_axis(pyvjoy.HID_USAGE_Y, 0x4000)
j.set_axis(pyvjoy.HID_USAGE_SL0, 0)
j.set_axis(pyvjoy.HID_USAGE_SL1, 0)

# Modify the IP and PORT according to the server
IP = 'the IP of the server'
PORT = 'the port of the server'
client.connect((IP, PORT))  # Connect to the server

while True:
    data = client.recv(1024)  # Receive data
    message = data.decode()
    message = message.strip()
    messages = message.split()
    count = len(messages)
    num = int(count/4)

    # Since the server's information publishing frequency may be higher than the client's information reception frequency,
    # the received messages may contain multiple sets of instructions. We need to process each set of instructions sequentially.
    for i in range(num):
        po_angle = messages[i*4]
        po_gear = messages[i*4+1]
        po_brake = messages[i*4+2]
        po_acce = messages[i*4+3]

        # If not in gear shifting mode, the steering wheel can be turned
        print(po_angle, po_gear, po_brake, po_acce)
        if po_gear == 'N':
            po_angle = int(po_angle)
            j.set_axis(pyvjoy.HID_USAGE_X, po_angle)

        # When shifting up or down, send joystick commands
        if po_gear == 'U':
            j.set_axis(pyvjoy.HID_USAGE_Y, 0x7FFF)
        elif po_gear == 'D':
            j.set_axis(pyvjoy.HID_USAGE_Y, 0x0000)
        # Reset the joystick position if not in gear shifting mode, gear shifting mode ends,
        # or if gear shifting mode is entered but no upshifting or downshifting operation is performed
        else:
            j.set_axis(pyvjoy.HID_USAGE_Y, 0x4000)

        # When braking, send trigger commands
        if po_brake != 'F':
            po_brake = int(po_brake)
            j.set_axis(pyvjoy.HID_USAGE_SL0, po_brake)
        # Reset the trigger if not in braking state or when braking stops
        else:
            j.set_axis(pyvjoy.HID_USAGE_SL0, 0x0000)

        # When accelerating, send trigger commands
        if po_acce != 'F':
            po_acce = int(po_acce)
            j.set_axis(pyvjoy.HID_USAGE_SL1, po_acce)
        # Reset the trigger if not in acceleration state or when acceleration stops
        else:
            j.set_axis(pyvjoy.HID_USAGE_SL1, 0x0000)

    time.sleep(0.017)

    if po_gear == 'N':
        po_angle = int(po_angle)
        j.set_axis(pyvjoy.HID_USAGE_X, po_angle+20)

    # When shifting up or down, send joystick commands
    if po_gear == 'U':
        j.set_axis(pyvjoy.HID_USAGE_Y, 0x7FFF)
    elif po_gear == 'D':
        j.set_axis(pyvjoy.HID_USAGE_Y, 0x0000)
    # Reset the joystick position if not in gear shifting mode, gear shifting mode ends,
    # or if gear shifting mode is entered but no upshifting or downshifting operation is performed
    else:
        j.set_axis(pyvjoy.HID_USAGE_Y, 0x4000)

    # When braking, send trigger commands
    if po_brake != 'F':
        po_brake = int(po_brake)
        j.set_axis(pyvjoy.HID_USAGE_SL0, po_brake)
    # Reset the trigger if not in braking state or when braking stops
    else:
        j.set_axis(pyvjoy.HID_USAGE_SL0, 0x0000)

    # When accelerating, send trigger commands
    if po_acce != 'F':
        po_acce = int(po_acce)
        j.set_axis(pyvjoy.HID_USAGE_SL1, po_acce)
    # Reset the trigger if not in acceleration state or when acceleration stops
    else:
        j.set_axis(pyvjoy.HID_USAGE_SL1, 0x0000)

client.close()