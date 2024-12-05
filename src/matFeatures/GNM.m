classdef GNM
    %{
        This class is used to compute the GNM modes not considering the environment.
        Example:
        --------
        % addpath('/improve/src/matFeatures');
        % importlib;
        pdbPath = '/project/getFeature/matFeatures/4zwj_opm_OPM.pdb';
        full     = readPDB(pdbPath);
        protein  = as('protein', full);
        ca       = as('name CA', protein);
        data = GNM(ca, 7.3);
        cutOff in the range of 7.3 to 15.0 A do not make much difference (Lee-Wei Yang et al,. 2006).
    %}
    properties
        system      {struct}    % CA or nucleotide
        cutOff      {double}    % Amstrong
        H           {double}    % Hessian matrix
        GVals       {double}    % N-1 eigenvalues of the Hessian
        GVecs       {double}    % N x (N-1) eigenvectors of the Hessian
    end

    methods
        %~ Constructor method
        function obj = GNM(system, cutOff)
            % Import libraries
            % addpath('/improve/src/matFeatures');
            % importlib;

            if ~exist('cutOff', 'var')
                cutOff = 7.3;
            end
            obj.cutOff = cutOff;
            obj.system = system;
            obj = obj.Hessian(obj.system, obj.cutOff);
            obj = obj.get(obj.H);

            % Run GNM until GVals and GVecs are not NaN or cutOff reaches 15.0
            while any(isnan(obj.GVals)) && any(isnan(obj.GVecs))
                obj.cutOff = obj.cutOff + 1;
                if obj.cutOff > 15
                    warning('cutOff reached 15.0 A. GVals and GVecs are still NaN.');
                    break;
                end

                % Run Hessian and get modes
                obj = obj.Hessian(obj.system, obj.cutOff);
                obj = obj.get(obj.H);
            end
        end

        function obj = Hessian(obj, system, cutOff)
            H = getGNMContactMatrix(system, cutOff); % ./core
            obj.H = H;
        end

        function obj = get(obj, H)
            [GVecs, GVals]  = eig(H);
            GVals           = diag(GVals);
            [GVals, idx]    = sort(GVals);
            GVecs           = GVecs(:, idx);
        
            if GVals(1) > 10^(-9)
                warningMessage = sprintf('The 1st eigenvalue is not zero for cutoff = %d A.', obj.cutOff);
                warning(warningMessage);
                obj.GVals = NaN;
                obj.GVecs = NaN;
            elseif GVals(2) < 10^(-9)
                warningMessage = sprintf('The 2nd eigenvalue is zero for cutoff = %d A.', obj.cutOff);
                warning(warningMessage);
                obj.GVals = NaN;
                obj.GVecs = NaN;
            else
                obj.GVals = GVals(2:end);
                obj.GVecs = GVecs(:, 2:end);
            end
        end
    end
    
    methods (Static)

        function [H] = getKirchhoff(kdat)
            %{
                Computes the H matrix as loading kdat file.
                Parameters:
                -----------
                    kdat: mat file
                        The kdat matrix file. temp.kdat
                
                Returns:
                --------
                    H: double, N x N
                        The H matrix of the system NOT considering the environment.
            %}
            H = load(kdat);
            H = spconvert(A);
            H = A + A';
            H = spdiags(1/2 * spdiags(H, 0), 0, A);
        end
    end
end