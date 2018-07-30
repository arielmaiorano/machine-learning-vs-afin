###############################################################################
# Generación del dataset
###############################################################################

# archivos de texto
ARCHIVOS_TEXTO = [
		 '../textos/texto_1_Wikipedia.com-Criptografía.txt',
		 '../textos/texto_2_Wikepedia.com-Criptoanálisis.txt',
		 '../textos/texto_3_JLB-Funes el memorioso.txt',
		 '../textos/texto_4_HPL-El clérigo malvado.txt',
		 '../textos/texto_5_Wikepedia.com-Argentina.txt'
		]

# archivo de salida
ARCHIVO_SALIDA = './dataset.csv'

# opciones
#AFIN_A = [1,3,5,7,9,11,15,17,19,21,23,25]
AFIN_A = [1]

AFIN_B = [3]

# longitud de mensaje fija
CARACTERES_FIX = 16

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
header = 'claro,cifrado'
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
				for a in AFIN_A:
					for b in AFIN_B:
						cifrado = Affine(a, b).encipher(claro)
						print(claro + ',' + cifrado, file=f)
						total_lineas = total_lineas + 1

print('Total de registros: ' + str(total_lineas))
print('Proceso finalizado.')
