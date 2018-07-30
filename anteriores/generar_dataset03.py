###############################################################################
# Generación del dataset
###############################################################################

# archivos de texto
ARCHIVOS_TEXTO = ['texto_1_Wikipedia.com-Criptografía.txt', 'texto_2_Wikepedia.com-Criptoanálisis.txt',
		'texto_3_JLB-Funes el memorioso.txt', 'texto_4_HPL-El clérigo malvado.txt',
		'texto_5_Wikepedia.com-Argentina.txt']
#ARCHIVOS_TEXTO = ['texto_3_JLB-Funes el memorioso.txt', 'texto_4_HPL-El clérigo malvado.txt']

# archivo de salida
ARCHIVO_SALIDA = 'dataset.csv'

# opciones
AFIN_A = [1]
#AFIN_B = [3,13,20]
#AFIN_B = [12,13,14]
#AFIN_B = [6,13,23]
#AFIN_B = [4,14,21]
AFIN_B = [7,14,24]

# longitud de mensaje fija
CARACTERES_FIX = 32

# caracteres permitidos (fijo)
CARACTERES_RE = '[^A-Z]'

###############################################################################

from pycipher import Affine
import re

print('Iniciando proceso...')

# caracters permitidos
regex = re.compile(CARACTERES_RE)

# abrir archivo reescribiendo cada vez
f = open(ARCHIVO_SALIDA, 'w')
header = ''
for i in range(CARACTERES_FIX):
	header = header + 'claro' + str(i+1) + ','
for i in range(CARACTERES_FIX):
	header = header + 'cifrado' + str(i+1) + ','
header = header + 'aff'
print(header, file=f)

total_lineas = 0

# procesar archivos de texto
for archivo in ARCHIVOS_TEXTO:
	with open(archivo, encoding="utf8") as f2:
		contenido = f2.readlines()
		for linea in contenido:
			linea = linea.strip().upper()
			if linea == '':
				continue
			limpio = regex.sub('', linea)
			for claro in re.findall('.{%d}' % CARACTERES_FIX, limpio):
				salida = ''
				lista_claro = list(claro)
				for caracter in lista_claro:
					salida = salida + str(ord(caracter)) + ','
				for a in AFIN_A:
					for b in AFIN_B:
						salida2 = ''
						cifrado = Affine(a, b).encipher(claro)
						lista_cifrado = list(cifrado)
						for caracter in lista_cifrado:
							salida2 = salida2 + str(ord(caracter)) + ','
						###aff = (a * 100) + b
						aff = b
						print(salida + salida2 + str(aff), file=f)
						
						# xxx
						#print(salida2 + salida + str(aff), file=f)
						
						total_lineas = total_lineas + 1

print('Total de registros: ' + str(total_lineas))
print('Proceso finalizado.')
