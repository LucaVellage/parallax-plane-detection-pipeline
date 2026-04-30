"""
Script for initial authentication and initialisation of Google Earth Engine
"""

import ee 
import os
from dotenv import load_dotenv

load_dotenv()
GEE_PROJECT = os.getenv('GEE_PROJECT')

def gee_auth_init():

    #print("Enter GEE token into search box + press Enter")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    print(f'GEE initialised successfully')
