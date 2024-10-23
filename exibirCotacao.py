import sys
import MetaTrader5 as mt5
import time

mt5.initialize()

while(True):
    ativo = mt5.symbol_info_tick('INDQ24')
    sys.stdout.write('\r' + str(ativo.last)) # \r apaga a cotação anterior
    sys.stdout.flush() # flush limpa a memória
    time.sleep(0.1)

    
