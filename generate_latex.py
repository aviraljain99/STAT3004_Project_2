N = 3

stateList = [(x, y) for x in range(N + 1) for y in range(N + 1) if x + y <= N]

represen = " & ".join([str(state)  for state in stateList])
represen = "state &" + represen + "\\\\" + "\n"

# print(represen)

for ind, state in enumerate(stateList, 0):
    elements = []
    for to in stateList:
        if to == state:
            elements.append("\\lambda_{{{0}}}".format(state))
        elif to == (state[0] - 1, state[1] + 1) and state[0] > 0:
            product = state[0] * state[1]
            if product == 0:
                elements.append("0")
            else:
                elements.append("${0}\\text{{R\\textsubscript{{SI}}}}$".format(product if product != 1 else ""))
        elif to == (state[0], state[1] - 1) and state[1] > 0:
            elements.append("${0}\\text{{R\\textsubscript{{IR}}}}$".format(state[1] if state[1] != 1 else "")) 
        else:
            elements.append("0")
    line = "{}".format(state) + " & " + " & ".join(elements) 
    if ind < len(stateList) - 1:
        line = line + "\\\\"
    represen += line + "\n"

print(represen) 
