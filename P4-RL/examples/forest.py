import mdptoolbox.example
P, R = mdptoolbox.example.forest()
print(P)
print(R)
Psp, Rsp = mdptoolbox.example.forest(is_sparse=True)
len(Psp)
print(Psp[0])
print(Psp[1])
print(Rsp)
(Psp[0].todense() == P[0]).all()
(Rsp == R).all()