import re
import os

def stringSplitByNumbers(x):
    '''
    from comment here
    http://code.activestate.com/recipes/135435-sort-a-string-using-numeric-order/
    '''
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]
    
