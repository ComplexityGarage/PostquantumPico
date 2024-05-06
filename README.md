# PostquantumPico

**Note:** The PostquantumPico project has an educational and experimental purpose. Not to be used in production.

The project aims to implement post-quantum cryptographic algorithms.

## Currently implemented algorithms:
### Digital signatures
- UOV (Unbalanced Oil and Vinegar)

## Installation
1. Install Micropython on your Raspberry Pi Pico according to the instructions: [Micropython installation on Raspberry Pi Pico](https://www.raspberrypi.com/documentation/microcontrollers/micropython.html)
2. Download the files: `PostquantumPico.py`, `pyboard.py` and `UOV.py`
3. Connect the Pico to a USB port
4. Run the following command in the terminal:
`python3 pyboard.py --device /dev/ttyACM0 -f cp UOV.py :`

## Operation
- Run `PostaquantumPico.py` file:
`python3 PostaquantumPico.py`
- The running console application allows the generation of a private key on the Pico and signing files with it