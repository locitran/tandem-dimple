

# Define a Python subroutine to colour atoms by B-factor, using predefined intervals
# Pathogenic probability (time 100) is represented in B-factor column 
# Visualize the pathogenicity of the protein structure using PyMOL
# Color scheme follows coolwarm gradient ranging from 0 low pathogenicity (blue) to 9 high pathogenicity (red)


from pymol import cmd

def colour_pathogenicity(selection="all"):
    """
    Colors a structure using a continuous coolwarm gradient based on B-factor
    (interpreted as pathogenicity probability × 100, ranging from 0 to 100).
    """

    # Step 1: Color other chains gray (but keep O/N/H colors)
    cmd.color("gray", selection)
    cmd.util.cnc()

    # Step 2: Define custom deep blue to deep red coolwarm endpoints
    cmd.set_color("coolblue",  [0.229, 0.298, 0.753])
    cmd.set_color("lightblue", [0.572, 0.739, 0.954])
    cmd.set_color("white",     [1.000, 1.000, 1.000])
    cmd.set_color("lightred",  [0.988, 0.705, 0.538])
    cmd.set_color("coolred",   [0.705, 0.015, 0.149])


    # Step 3: Apply spectrum coloring based on B-factor
    cmd.spectrum("b", "coolblue lightblue white lightred coolred", selection=selection, minimum=0, maximum=100)

    # Step 4: Final display
    cmd.hide("everything", "hetatm")
    cmd.show("cartoon", selection)
    cmd.deselect()

    # Grey colour for B-factor -1
    cmd.set_color("bfactor_gray", [0.6, 0.6, 0.6])  # light gray
    cmd.select("bfactor_zero", selection + " & b = -1")
    cmd.color("bfactor_gray", "bfactor_zero")
    cmd.set("cartoon_transparency", 0.8, "bfactor_zero")

# Register the command
cmd.extend("colour_pathogenicity", colour_pathogenicity)