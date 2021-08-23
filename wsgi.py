import os, sys
path = os.path.abspath(__file__)
dirname = os.path.dirname(path)
sys.path.insert(0, dirname)
##
from interval import app
application = app.server

