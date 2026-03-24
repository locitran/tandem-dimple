from pymol import cmd

def make_balls(selection="all",
               residues=[1, 2, 3],
               size=1.0,
               ):
    """
    Visualize CA atoms as spheres with specific radius and color.

    pathogenic: 3.5 A
    benign: 2.5 A
    middle: 3.0 A

    """
    # Select target CA atoms for given {residues}
    for resi in residues:
        sel_name = f"ball_resi_{resi}"
        cmd.select(sel_name, f"{selection} and name CA and resi {resi}")
        cmd.show("spheres", sel_name)
        cmd.set("sphere_scale", size, sel_name)
        cmd.delete(sel_name)

# Register the command
cmd.extend("make_balls", make_balls)


"""
make_balls('chainA', residues=[ 34,  37,  44,  44,  50,  59,  75,  75,  84,  90,  95, 143, 143, 161, 163, 179, 184, 195, 202, 205, 206],size=2)

make_balls('chainA', residues=[217, 215, 214, 210, 203, 170, 170, 168, 156, 153, 127, 123, 121, 115, 114, 111, 107, 100,  83,  27,  16,   4,   4, 165],size=1)

make_balls('chainA', residues=[197],size=1.5)

make_balls('chainA_consurf', residues=[ 34,  37,  44,  44,  50,  59,  75,  75,  84,  90,  95, 143, 143, 161, 163, 179, 184, 195, 202, 205, 206],size=2)

make_balls('chainA_consurf', residues=[217, 215, 214, 210, 203, 170, 170, 168, 156, 153, 127, 123, 121, 115, 114, 111, 107, 100,  83,  27,  16,   4,   4, 165],size=1)

make_balls('chainA_consurf', residues=[197],size=1.5)

make_balls('chainA_V1', residues=[ 34,  37,  44,  44,  50,  59,  75,  75,  84,  90,  95, 143, 143, 161, 163, 179, 184, 195, 202, 205, 206],size=2)

make_balls('chainA_V1', residues=[217, 215, 214, 210, 203, 170, 170, 168, 156, 153, 127, 123, 121, 115, 114, 111, 107, 100,  83,  27,  16,   4,   4, 165],size=1)

make_balls('chainA_V1', residues=[197],size=1.5)



[217, 215, 214, 210, 203, 170, 170, 168, 156, 153, 127, 123, 121, 115, 114, 111, 107, 100,  83,  27,  16,   4,   4, 165]
"""