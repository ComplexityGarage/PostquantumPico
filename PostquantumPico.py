import pyboard

print(
'''
===================
|                 |
|    PicoCrypto   |
|                 |
===================
''')

def sign(file_name,new_key=False):
    cmd = '''
n = 160//16
m = 64//16

from UOV import *

uov = UOV(m,n)
with open('priv_key','rb') as file:
    sk_seed = file.read()

pk_seed, P3s = uov.CompactKeysGen(sk_seed)

with open('{file_name}', 'rb') as file:
    M = file.read()

s, salt = uov.Sign(M)

uov.PublicOutputToFile(salt, pk_seed, s, P3s)
    '''.format(file_name=file_name)
    return cmd

def keygen():
    cmd='''
n = 160//16
m = 64//16

from UOV import *

uov = UOV(m,n)

pk_seed, P3s = uov.CompactKeysGen()
uov.PrivateKeyToFile()
'''
    return cmd

PRIV_KEY_FILE = 'priv_key'
SIGNATURE_FILE = 'signature'

try:
    pyb = pyboard.Pyboard('/dev/ttyACM0',115200)
    pyb.enter_raw_repl()
except:
    print("Nie znaleziono PicoCrypto, najpierw podłącz urządzenie.")
else:

    menu = input('Wybierz działanie (1/2/3/4).\n1. Podpisz plik\n2. Wygeneruj nowy klucz\n3. Skasuj istniejący klucz\n4. Wyjdź\n')
    if menu=='1':
        key_exist = pyb.fs_exists(PRIV_KEY_FILE)
        if key_exist:
            while True:
                file_name = input('Podaj nazwę pliku: ')
                try:
                    file = open(file_name,'rb')
                except:
                    print('Nie ma takiego pliku')
                else:
                    break
            pyb.fs_put(file_name,':')
            pyb.exec(sign(file_name))
            pyb.fs_get(SIGNATURE_FILE,SIGNATURE_FILE)
            print('Podpis zapisano w pliku "'+SIGNATURE_FILE+'"')
        else:
            new_key = input('Nie znaleziono klucza prywatnego. Czy wygenerować nowy? (t/n): ')
            if new_key=='t':
                pyb.exec(keygen())
                print('Klucz wygenerowano')
            else:
                print('Koniec programu')
    elif menu=='2':
        pyb.exec(keygen())
        print('Klucz wygenerowano')
    elif menu=='3':
        try:
            pyb.fs_rm(PRIV_KEY_FILE)
            print('Klucz skasowano')
        except:
            print('Nie znaleziono klucza prywatnego')
    elif menu=='4':
        print('Koniec programu')
    pyb.exit_raw_repl()
    pyb.close()