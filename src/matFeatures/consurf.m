classdef getConSurf
    properties
        pdbID           {char}  
        system          {struct} % Contain C-alpha atoms
        chainID         {struct}
    end

    methods
        function obj = getConSurf(pdbID, custom_PDB)
            % Import libraries
            % addpath('/project/src/matFeatures');
            % importlib;
            obj.pdbID = pdbID;

            % Read PDB file
            if ~exist('custom_PDB', 'var')
                % Parse pdb file from RCSB based on pdbID
                pdbPath = sprintf('../pdbfile/download/%s.pdb', pdbID);
                % Download pdb file from RCSB
                if ~exist(pdbPath, 'file')
                    url = sprintf('https://files.rcsb.org/download/%s.pdb', pdbID);
                    websave(pdbPath, url);
                end
            else
                pdbPath = custom_PDB;
            end

            % Read PDB file
            pdb = getFeatures.parsePDB(pdbPath, True);
        end
