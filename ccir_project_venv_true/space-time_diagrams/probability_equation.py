
psi = 1/3
phi = 2/3

# P = (psi*phi)^2/phi*phi

#markov chain - psi*phi/1 + psi*phi -->


expr = ((psi*phi)**2)/phi*phi + ((psi*((phi*phi)**2))**2)/(psi*phi)

print(expr)


expr2 = ((psi*phi)**2)/phi*phi + ((psi*(phi*phi))**2)/(psi*phi) + ((psi*(psi*phi))**2)/(psi*phi*phi)

print(expr2)


expr3 = ((psi*phi)**2)/phi*phi + ((psi*(phi*phi))**2)/(psi*phi) + ((psi*(psi*phi))**2)/(psi*phi*phi) + ((psi*(psi*phi*phi))**2)/(psi*(psi*phi))

print(expr3)





