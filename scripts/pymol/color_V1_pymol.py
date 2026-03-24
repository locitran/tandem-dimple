

# Define a Python subroutine to colour atoms by B-factor, using predefined intervals
# V1: first mode from GNM
# Value of V1 is min-max normalized in range of 0 to 100 and saved in B-factor column
# Color scheme follows white (0) to deep red (100) gradient

from pymol import cmd

def colour_V1(selection="all"):

    # Step 1: Color other chains gray (but keep O/N/H colors)
    cmd.color("gray", selection)
    cmd.util.cnc()

    # Step 2: Define custom deep blue to deep red coolwarm endpoints
    cmd.set_color("white",     [1.000, 1.000, 1.000])
    cmd.set_color("lightred",  [0.988, 0.705, 0.538])
    cmd.set_color("coolred",   [0.705, 0.015, 0.149])

    # Step 3: Apply spectrum coloring based on B-factor
    cmd.spectrum("b", "white lightred coolred", selection=selection, minimum=0, maximum=100)

    # Step 4: Final display
    cmd.hide("everything", "hetatm")
    cmd.show("cartoon", selection)
    cmd.deselect()

    # Grey colour for B-factor -1
    cmd.set_color("bfactor_gray", [0.6, 0.6, 0.6])  # light gray
    cmd.select("bfactor_zero", selection + " & b = -1")
    cmd.color("bfactor_gray", "bfactor_zero")
    cmd.set("cartoon_transparency", 0.5, "bfactor_zero")

# Register the command
cmd.extend("colour_pathogenicity", colour_V1)