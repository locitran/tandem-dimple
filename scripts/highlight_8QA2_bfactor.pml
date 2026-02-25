reinitialize
load /home/loci/main/tandem_website_dev/tandem/data/GJB2/structures/8QA2.pdb, prot

# Clean base view
hide everything, all
show cartoon, prot
bg_color white

# Selections
select focus_chains, prot and chain A+G
select other_chains, prot and not chain A+G

# De-emphasize other chains
color gray85, other_chains
set cartoon_transparency, 0.80, other_chains

# Highlight chain A and G atoms, color by B-factor
spectrum b, blue_white_red, focus_chains

spectrum b, white_red, focus_chains, minimum=0, maximum=100

# Optional polish
set cartoon_fancy_helices, 1
set antialias, 2
set ray_opaque_background, off
orient focus_chains
zoom focus_chains, 8


select focus_chains, 2ZW3 and chain A+D-2
select other_chains, 2ZW3 and not chain A+D-2
color gray85, other_chains
set cartoon_transparency, 0.80, other_chains
