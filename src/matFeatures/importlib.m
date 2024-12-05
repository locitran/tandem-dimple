function importlib()
    % Get dir of this file
    [pathstr,~,~] = fileparts(mfilename('fullpath'));
    addpath([pathstr '/bioStructureM/core']);
    addpath([pathstr '/bioStructureM/atomselector']);
    addpath([pathstr '/bioStructureM/accelerator']);
    if ismac
        % Code to run on Mac platform
    elseif isunix
        % Code to run on Linux platform
    elseif ispc
        % Code to run on Windows platform
    else
        disp('Platform not supported')
    end
end