###############################################################################
# Generación del dataset
###############################################################################


# archivo de salida
ARCHIVO_SALIDA = './dataset.csv'

# cantidad de variables (estados anteriores)
CANTIDAD_BYTES_CONOCDIDOS = 5

# parámetros para lfsr
LFSR_ESTADO_INICIAL = '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001' # '000000000000000000001'
#LFSR_TAPS = (21,19)
LFSR_TAPS = (168,166,153,15)

### LFSR_TOTAL_BITS_A_GENERAR = 9999999999 * 8 * (CANTIDAD_BYTES_CONOCDIDOS + 1)


###############################################################################


import re
import unidecode


###############################################################################
def lfsr(seed, taps):
	sr, xor = seed, 0
	while 1:
		for t in taps:
			xor += int(sr[t-1])
		if xor%2 == 0.0:
			xor = 0
		else:
			xor = 1
		sr, xor = str(xor) + sr[:-1], 0
		yield sr
		if sr == seed:
			break
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
for i, estado in enumerate(lfsr(LFSR_ESTADO_INICIAL, LFSR_TAPS)):

	if i % 8 == 0 and i > 0:

		idx = (j % (CANTIDAD_BYTES_CONOCDIDOS + 1))
		if idx == 0 and j > 0:
			print(tmp_linea[:-1], file=f)
			total_lineas = total_lineas + 1
			
			#xxx
			if total_lineas > 50000:
				break

			tmp_linea = ''

		tmp_linea += str(int(byte_salida, 2)) + ','

		j += 1

		byte_salida = ''

	byte_salida += estado[0]

	#if i == LFSR_TOTAL_BITS_A_GENERAR:
	#	break
	bits += 1

print('Total de registros: ' + str(total_lineas))
print('Proceso finalizado.')
print()
print('bits: ' + str(bits))
