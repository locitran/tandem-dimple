import sys
from chimerax.core.commands import run as rc

def colour_consurf(filename, output_file):
    rc(session, "open " + filename)
    # Colour other chains gray, while maintaining
    # oxygen in red, nitrogen in blue and hydrogen in white
    rc(session, "color name consgray 50,50,50")
    #rc(session, "color consgray All")
    #rc(session, "color byhetero consgray,r")
    #rc(session, "color byhet")
    
    # Colours are calculated by dividing the RGB colours by 255
    # RGB = [[16,200,209],[140,255,255],[215,255,255],[234,255,255],[255,255,255],
    #        [252,237,244],[250,201,222],[240,125,171],[160,37,96]]
    #rc(session, "color name cons10 100,100,59")
    rc(session, "color name cons9 47,16,51")
    rc(session, "color name cons8 61,43,67")
    rc(session, "color name cons7 76,65,80")
    rc(session, "color name cons6 90,82,90")
    rc(session, "color name cons5 100,100,100")
    rc(session, "color name cons4 84,94,82")
    rc(session, "color name cons3 65,86,63")
    rc(session, "color name cons2 35,69,37")
    rc(session, "color name cons1 6,35,14")
    rc(session, "color name cons10 100,100,59")

    
    # Colour by bfactor    
    rc(session, "color by bfactor palette 0,consgray:10,cons10:1,cons1:2,cons2:3,cons3:4,cons4:5,cons5:6,cons6:7,cons7:8,cons8:9,cons9")
    
    # Present in a publication view and save Chimera session
    #rc(session, "preset apply pub 3")
    #rc(session, "focus")
    rc(session, "hide atoms")
    rc(session, "show cartoon")
    rc(session, "show ligand")
    rc(session, "style ligand sphere")
    rc(session, "save " + output_file)

colour_consurf(sys.argv[1], sys.argv[2])