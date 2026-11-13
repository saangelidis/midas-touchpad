#
# Keyboard Output - FOR LINUX
#

import pickle
from evdev import UInput, ecodes as e
import time
import numpy as np


def typeCharacter(character):
    pass


def main():
    mathOperators = np.array([
    r"\sum", r"\partial", r"\int", r"\cdot", "-", "+",
    r"\amalg", r"\cup", r"\oplus", r"\times", r"\ast",
    r"\triangleleft", r"\otimes", r"\triangleright", r"\diamond", r"\pm",
    r"\div", r"\bullet", r"\setminus", r"\uplus", r"\cap",
    r"\mp", r"\sqcap", r"\vee", r"\odot", r"\sqcup",
    r"\wedge", r"\circ", r"\ominus", r"\star", r"\wr",
    r"\barwedge", r"\circledcirc", r"\boxdot", r"\ltimes", r"\boxplus",
    r"\boxtimes", r"\rtimes", r"\circledast", r"\lhd", r"\prod",
    r"\coprod", r"\oint", r"\parr", r"\with", r"\fint",
    r"\varoiint", r"\oiint", r"\nabla", r"\neg", r"\sqrt{}",
    ])

    mathUnicode = np.array([
    0x2211, 0x2202, 0x222B, 0x22C5, 0x002D, 0x002B,   
    0x2A3F, 0x222A, 0x2295, 0x00D7, 0x2217,           
    0x25C1, 0x2297, 0x25B7, 0x22C4, 0x00B1,           
    0x00F7, 0x2219, 0x2216, 0x228E, 0x2229,           
    0x2213, 0x2293, 0x2228, 0x2299, 0x2294,           
    0x2227, 0x2218, 0x2296, 0x22C6, 0x2240,           
    0x22BC, 0x229A, 0x22A1, 0x22C9, 0x229E,           
    0x22A0, 0x22CA, 0x229B, 0x22B2, 0x220F,           
    0x2210, 0x222E, 0x214B, 0x0026, 0x2A0F,           
    0x222F, 0x222F, 0x2207, 0x00AC, 0x221A,           
    ], dtype=np.uint32)

    mathMap = np.vstack((mathOperators,mathUnicode)).T

# print(mathMap.shape)
# print(mathMap)

    currentChar = r"\sum"

    for i in range(mathmap.shape[0]):
        if mathMap[i][0] == currentChar:
            typeCharacter(mathmap[i][1])

main()