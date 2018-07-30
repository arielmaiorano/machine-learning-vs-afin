###############################################################################
# Generación del dataset
###############################################################################

# archivos de texto
ARCHIVOS_TEXTO = [
		 #  '../textos/texto_1_Wikipedia.com-Criptografía.txt' 
		 #, '../textos/texto_2_Wikepedia.com-Criptoanálisis.txt'
		 '../textos/texto_3_JLB-Funes el memorioso.txt'
		 #, '../textos/texto_4_HPL-El clérigo malvado.txt'
		 #, '../textos/texto_5_Wikepedia.com-Argentina.txt'
		]

# archivo de salida
ARCHIVO_SALIDA = './dataset.csv'

# opciones
AFIN_A = [1,3,5,7,9,11,15,17,19,21,23,25]

AFIN_B = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

# longitud de mensaje fija
CARACTERES_FIX = 32

# caracteres permitidos (fijo)
CARACTERES_RE = '[^A-Z]'

###############################################################################

from pycipher import Affine
import re
import unidecode


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
			limpio = regex.sub('', unidecode.unidecode(linea))
			for claro in re.findall('.{%d}' % CARACTERES_FIX, limpio):
				salida = ''
				lista_claro = list(claro)
				for caracter in lista_claro:
					salida = salida + str(ord(caracter) - ord('A')) + ','
				for a in AFIN_A:
					for b in AFIN_B:
						salida2 = ''
						cifrado = Affine(a, b).encipher(claro)
						lista_cifrado = list(cifrado)
						for caracter in lista_cifrado:
							salida2 = salida2 + str(ord(caracter) - ord('A')) + ','
						aff = (a * 100) + b
						###aff = b
						print(salida + salida2 + str(aff), file=f)
						total_lineas = total_lineas + 1

print('Total de registros: ' + str(total_lineas))
print('Proceso finalizado.')
