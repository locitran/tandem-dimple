from pymol import cmd

def colour_consurf(selection="all"):
    """Colour atoms by ConSurf values already stored in the B-factor column."""
    cmd.color("gray", selection)
    cmd.util.cnc()

    minimum = 0.0
    maximum = 9.0
    n_colours = 9
    colours = [
        [0.039215686, 0.490196078, 0.509803922],
        [0.294117647, 0.68627451, 0.745098039],
        [0.647058824, 0.862745098, 0.901960784],
        [0.843137255, 0.941176471, 0.941176471],
        [1, 1, 1],
        [0.980392157, 0.921568627, 0.960784314],
        [0.980392157, 0.784313725, 0.862745098],
        [0.941176471, 0.490196078, 0.666666667],
        [0.62745098, 0.156862745, 0.37254902],
    ]
    bin_size = (maximum - minimum) / n_colours

    for i in range(n_colours):
        lower = minimum + i * bin_size
        upper = lower + bin_size
        group = selection + "_group_" + str(i + 1)
        colour_name = "colour_" + str(i + 1)

        if i == n_colours - 1:
            sel_string = selection + " & ! b < " + str(lower) + " & ! b > " + str(maximum)
        else:
            sel_string = selection + " & ! b < " + str(lower) + " & b < " + str(upper)

        cmd.select(group, sel_string)
        cmd.set_color(colour_name, colours[i])
        cmd.color(colour_name, group)

    insuf_colour = [1, 1, 0.588235294]
    cmd.set_color("insufficient_colour", insuf_colour)
    cmd.select("insufficient", selection + " & b = 10")
    cmd.color("insufficient_colour", "insufficient")

    cmd.set_color("bfactor_zero_gray", [0.6, 0.6, 0.6])
    cmd.select("bfactor_zero", selection + " & b = -1")
    cmd.color("bfactor_zero_gray", "bfactor_zero")
    cmd.set("cartoon_transparency", 0.5, "bfactor_zero")

    cmd.hide("everything", "hetatm")
    cmd.show("cartoon", selection)
    cmd.deselect()


def save_consurf_session(pdbfile, outfile="consurf.pse", obj_name="prot"):
    """Load a patched PDB, apply ConSurf colouring, and save a PyMOL session."""
    cmd.load(pdbfile, obj_name)
    colour_consurf(obj_name)
    cmd.save(outfile)


cmd.extend("colour_consurf", colour_consurf)
cmd.extend("save_consurf_session", save_consurf_session)
