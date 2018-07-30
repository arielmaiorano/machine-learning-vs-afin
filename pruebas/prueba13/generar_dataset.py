###############################################################################
# Generación del dataset
###############################################################################


# archivo de salida
ARCHIVO_SALIDA = './dataset.csv'

# cantidad de variables (estados anteriores)
CANTIDAD_BYTES_CONOCDIDOS = 5

TOTAL_BYTES_A_GENERAR = 200000 * 8 * 6


###############################################################################


import re
import unidecode


# RC4 #########################################################################
def KSA(key):
    keylength = len(key)
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + key[i % keylength]) % 256
        S[i], S[j] = S[j], S[i]  # swap
    return S
def PRGA(S):
    i = 0
    j = 0
    while True:
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]  # swap
        K = S[(S[i] + S[j]) % 256]
        yield K
def RC4(key):
    S = KSA(key)
    return PRGA(S)
def convert_key(s):
    return [ord(c) for c in s]
###############################################################################

print('Iniciando proceso...')

# abrir archivo reescribiendo cada vez
f = open(ARCHIVO_SALIDA, 'w')
header = ''
for i in range(CANTIDAD_BYTES_CONOCDIDOS):
	header = header + 'b' + str(i) + ','
header = header + 'b' + str(CANTIDAD_BYTES_CONOCDIDOS)
print(header, file=f)

total_lineas = 0
tmp_linea = ''
tmp_byte = ''
j = 0
byte_salida = ''
bits = 0


llave = [0 for x in range(16)]
keystream = RC4(llave)

while bits * 8 < TOTAL_BYTES_A_GENERAR:

	byte_salida = keystream.__next__()
	idx = (j % (CANTIDAD_BYTES_CONOCDIDOS + 1))
	if idx == 0 and j > 0:
		print(tmp_linea[:-1], file=f)
		total_lineas = total_lineas + 1
		tmp_linea = ''

	tmp_linea += str(int(byte_salida)) + ','

	j += 1
	bits += 8


print('Total de registros: ' + str(total_lineas))
print('Proceso finalizado.')
print()
print('bits: ' + str(bits))
