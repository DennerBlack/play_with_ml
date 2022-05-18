weights = [0.3, 0.2, 0.9]

def ele_mul(number,vector):
    output = [0, 0, 0]
    assert(len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output


def nn(inp,wei):
    pred = ele_mul(inp, wei)
    return pred


wlrec = [0.65, 0.8, 0.8, 0.9]
input = wlrec[0]
pred = nn(input, weights)

print(pred)