weights = [0.1, 0.2, 0]

def w_sum(inp, wei):
    assert(len(inp) == len(wei))
    output = 0
    for i, j in zip(inp, wei):
        output += (i*j)
    return output

def nn(inp,wei):
    pred = w_sum(inp, wei)
    return pred

toes =  [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]


input = [toes[0],wlrec[0],nfans[0]]
pred = nn(input,weights)

print(pred)
