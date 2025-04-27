#!/bin/csh
if ($#argv == 0) then
	echo "USAGE: tagR4S.sh <version number>"
else
	set v = $argv[1];
	mkdir tags/ConSurf.v$v;
	svn add tags/ConSurf.v$v;
	svn cp trunk/Makefile tags/ConSurf.v$v/;
	mkdir tags/ConSurf.v$v/libs;
	svn add tags/ConSurf.v$v/libs;
	svn cp trunk/libs/Makefile tags/ConSurf.v$v/libs/;
	svn cp trunk/libs/phylogeny tags/ConSurf.v$v/libs/;
	mkdir tags/ConSurf.v$v/programs;
	svn add tags/ConSurf.v$v/programs;
	svn cp trunk/programs/Makefile tags/ConSurf.v$v/programs/;
	svn cp trunk/programs/Makefile.generic tags/ConSurf.v$v/programs/;
	svn cp trunk/programs/rate4site tags/ConSurf.v$v/programs/;
	mkdir tags/ConSurf.v$v/www;
	svn add tags/ConSurf.v$v/www;
	svn cp trunk/www/bioSequence_scripts_and_constants tags/ConSurf.v$v/www/;
	svn cp trunk/www/consurf tags/ConSurf.v$v/www/;
	svn cp trunk/www/Selecton tags/ConSurf.v$v/www/;
endif

# Don't forget to remove irrelevant libs and programs from the makefiles 
