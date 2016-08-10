#Copyright (C) Nial Peters 2015
#
#This file is part of gns_flyspec.
#
#gns_flyspec is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#gns_flyspec is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with gns_flyspec.  If not, see <http://www.gnu.org/licenses/>.

import main_script
import configuration
import sys
import os

def main():
        
    if len(sys.argv) == 1:
        print "No image file supplied"
        sys.exit()
    
    if len(sys.argv) > 2:
        print "You can only open one png file at a time"
        sys.exit()
    
    if not os.path.exists(sys.argv[1]):
        print "Cannot open %s. No such file."%sys.argv[1]
        sys.exit()
    
    if not sys.argv[1].endswith(('.png','.PNG')):
        print "Cannot open %s. Not a png file (is the file extension correct?)."%sys.argv[1]
        sys.exit()
    
    config = configuration.load_config()
    
    main_script.ScanSummaryFrame(config, False, True, from_png_file=sys.argv[1])
    
if __name__ == "__main__":
    main()
        
    