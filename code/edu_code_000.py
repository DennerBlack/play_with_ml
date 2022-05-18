weight = 0.1

def neur_net(input, weight):
    pred = input * weight
    return pred

games = [8, 9, 10, 9]

input = games[0]
pred = neur_net(input, weight)
print(pred)