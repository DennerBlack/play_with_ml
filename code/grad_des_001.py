wei = 0.1
lr = 0.01

def nn(inp, wei):
    pred = inp*wei
    return pred

nof = [8.5]
worl = [1]

inp = nof[0]
true = worl[0]

pred = nn(inp, wei)

err = (pred - true)**2
print(err)

p_up = nn(inp,wei+lr)
e_up = (p_up - true) ** 2
print(e_up)

p_dn = nn(inp,wei-lr)
e_dn = (p_dn - true) ** 2
print(e_dn)