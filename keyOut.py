#
# Keyboard Output - FOR LINUX
#

import pickle
from evdev import UInput, ecodes as e
import time
import numpy as np


def typeCharacter(keyboard : UInput, charID : hex) -> None:
    """
    Writes the character of 'charID' at the typespace.

    keyboard : UInput object - the virtual keyboard
    charID   : the character id which python sees as integer and has to be turned to a string

    """
    
    # "press" CTRL + SHIFT + U keys
    for k in ["KEY_LEFTCTRL", "KEY_LEFTSHIFT", "KEY_U"]:
        keyboard.write(e.EV_KEY, e.ecodes[k], 1); keyboard.syn()
    
    # "let go" of U, THEN SHIFT, THEN CTRL
    for k in ["KEY_U", "KEY_LEFTSHIFT", "KEY_LEFTCTRL"]:
        keyboard.write(e.EV_KEY, e.ecodes[k], 0); keyboard.syn()
    
    # "press" & "let go" of hex ID keys
    for k in ["KEY_" + c.upper() for c in charID] + ["KEY_SPACE"]:
        keyboard.write(e.EV_KEY, e.ecodes[k], 1); keyboard.syn()
        keyboard.write(e.EV_KEY, e.ecodes[k], 0); keyboard.syn()
        
        # small buffer
        time.sleep(0.01)



def keyOut(predictedSymbol, model_input):
    '''
    Converts the model's predicted LaTeX symbol into corresponding Unicode.
    Passes into typeCharacter for typespace output.

    predictedSymbol: str (LaTeX)
    model_input: str ("1" for Math, "2" for Greek"; this will be updated with the final submission's form of the user choosing the model characters subset.)
    '''

    mathMap = {
    r"\sum":             "2211",  # ∑
    r"\partial":         "2202",  # ∂
    r"\int":             "222B",  # ∫
    r"\cdot":            "22C5",  # ⋅
    "-":                 "002D",  # -
    "+":                 "002B",  # +
    r"\amalg":           "2A3F",  # ⨿
    r"\cup":             "222A",  # ∪
    r"\oplus":           "2295",  # ⊕
    r"\times":           "00D7",  # ×
    r"\ast":             "2217",  # ∗
    r"\triangleleft":    "25C1",  # ◁
    r"\otimes":          "2297",  # ⊗
    r"\triangleright":   "25B7",  # ▷
    r"\diamond":         "22C4",  # ⋄
    r"\pm":              "00B1",  # ±
    r"\div":             "00F7",  # ÷
    r"\bullet":          "2219",  # ∙
    r"\setminus":        "2216",  # ∖
    r"\uplus":           "228E",  # ⊎
    r"\cap":             "2229",  # ∩
    r"\mp":              "2213",  # ∓
    r"\sqcap":           "2293",  # ⊓
    r"\vee":             "2228",  # ∨
    r"\odot":            "2299",  # ⊙
    r"\sqcup":           "2294",  # ⊔
    r"\wedge":           "2227",  # ∧
    r"\circ":            "2218",  # ∘
    r"\ominus":          "2296",  # ⊖
    r"\star":            "22C6",  # ⋆
    r"\wr":              "2240",  # ≀
    r"\barwedge":        "22BC",  # ⊼
    r"\circledcirc":     "229A",  # ⊚
    r"\boxdot":          "22A1",  # ⊡
    r"\ltimes":          "22C9",  # ⋉
    r"\boxplus":         "229E",  # ⊞
    r"\boxtimes":        "22A0",  # ⊠
    r"\rtimes":          "22CA",  # ⋊
    r"\circledast":      "229B",  # ⊛
    r"\lhd":             "22B2",  # ⊲
    r"\prod":            "220F",  # ∏
    r"\coprod":          "2210",  # ∐
    r"\oint":            "222E",  # ∮
    r"\parr":            "214B",  # ⅋
    r"\with":            "0026",  # &
    r"\fint":            "2A0F",  # ⨏
    r"\varoiint":        "222F",  # ∯
    r"\oiint":           "222F",  # ∯
    r"\nabla":           "2207",  # ∇
    r"\neg":             "00AC",  # ¬
    r"\sqrt{}":          "221A",  # √
    }

    greekChars = {
    r"\alpha":      "03B1",  # α
    r"\beta":       "03B2",  # β
    r"\gamma":      "03B3",  # γ
    r"\Gamma":      "0393",  # Γ
    r"\delta":      "03B4",  # δ
    r"\Delta":      "0394",  # Δ
    r"\epsilon":    "03F5",  # ϵ
    r"\varepsilon": "03B5",  # ε
    r"\zeta":       "03B6",  # ζ
    r"\eta":        "03B7",  # η
    r"\theta":      "03B8",  # θ
    r"\vartheta":   "03D1",  # ϑ
    r"\Theta":      "0398",  # Θ
    r"\iota":       "03B9",  # ι
    r"\kappa":      "03BA",  # κ
    r"\varkappa":   "03F0",  # ϰ
    r"\lambda":     "03BB",  # λ
    r"\Lambda":     "039B",  # Λ
    r"\mu":         "03BC",  # μ
    r"\nu":         "03BD",  # ν
    r"\xi":         "03BE",  # ξ
    r"\Xi":         "039E",  # Ξ
    r"\pi":         "03C0",  # π
    r"\varpi":      "03D6",  # ϖ
    r"\Pi":         "03A0",  # Π
    r"\rho":        "03C1",  # ρ
    r"\varrho":     "03F1",  # ϱ
    r"\sigma":      "03C3",  # σ
    r"\Sigma":      "03A3",  # Σ
    r"\tau":        "03C4",  # τ
    r"\phi":        "03D5",  # ϕ
    r"\varphi":     "03C6",  # φ
    r"\Phi":        "03A6",  # Φ
    r"\chi":        "03C7",  # χ
    r"\psi":        "03C8",  # ψ
    r"\Psi":        "03A8",  # Ψ
    r"\omega":      "03C9",  # ω
    r"\Omega":      "03A9",  # Ω
    }

    # Select the appropriate Unicode character mapping
    if model_input == "1":
        charMap = mathMap
    else:
        charMap = greekChars

    # Convert the CNN's predicted LaTeX symbol to its Unicode hex code
    charNum = charMap.get(predictedSymbol)

    if charNum is None:
        print(f"No Unicode mapping found for {predictedSymbol}")
        return

    print(f"Predicted LaTeX: {predictedSymbol}")
    print(f"Unicode hex: {charNum}")

    # Initialize the Linux virtual keyboard
    with UInput() as kb:
        # wait for Wayland compositor to recognize it
        time.sleep(3)
        # type character through the virtual keyboard
        typeCharacter(kb, charNum)

