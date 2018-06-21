#!/bin/env python
# Save this as "url_encoding.py"	  
import sys
from urllib.parse import quote_plus

for l in sys.stdin:
	print(quote_plus(l.strip()) + '\n')