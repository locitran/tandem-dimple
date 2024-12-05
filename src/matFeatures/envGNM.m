classdef envGNM
    %{
        This class is used to compute the GNM modes considering the environment.
        Example:
        --------
        pdbPath = '/project/getFeature/matFeatures/4zwj_opm_OPM.pdb';
        full     = readPDB(pdbPath);
        protein  = as('protein', full);
        ca       = as('name CA and protein', protein);
        membrane = as('resn NE1 and name Q1', full);
        data = envGNM(ca, membrane, 7.3);

        [GVecs, GVals] = envGNM.getModes('temp_H_bar.mat', '.');
    %}
    properties
        system      {struct}    % Can be CA atoms
        environment {struct}    % Can be membrane molecules
        cutOff      {double}    % Amstrong
        HBar        {double}    % Hessian considering the environment
        GVals       {double}    % N-1 eigenvalues of the Hessian
        GVecs       {double}    % N x (N-1) eigenvectors of the Hessian
    end

    methods
        %~ Constructor method
        function obj = envGNM(system, environment, cutOff)
            % Import libraries
            % addpath('/project/src/matFeatures');
            % importlib;

            if ~exist('cutOff', 'var')
                cutOff = 7.3;
            end
            obj.system = system;
            obj.environment = environment;
            obj.cutOff = cutOff;
            obj = obj.Hessian(obj.system, obj.environment, obj.cutOff);
            obj = obj.get(obj.HBar);

            % Run GNM until GVals and GVecs are not NaN or cutOff reaches 15.0
            while any(isnan(obj.GVals)) && any(isnan(obj.GVecs))
                obj.cutOff = obj.cutOff + 0.1;
                if obj.cutOff >= 15.0
                    warning('cutOff reached 15.0 A. GVals and GVecs are still NaN.');
                    break;
                end

                % Run Hessian and get modes
                obj = obj.Hessian(obj.system, obj.environment, obj.cutOff);
                obj = obj.get(obj.HBar);
            end
        end

        function obj = Hessian(obj, system, environment, cutOff)
            %{
                normal GNM Hassian is like            
                |  Hss   Hse |
                |  Hse'  Hee |
                HBar =  Hss - Hse * Hee^-1 * Hse';

                Parameters:
                -----------
                    system: struct
                        The system can be a single chain or a set of chains, a protein or a complex.
                        The system coordinates are retrieved by system.coord
                    environment: struct
                        The environment can be a single chain or a set of chains, a protein or a complex.
                        The environment coordinates are retrieved by environment.coord
                    cutOff: double, optional
                        The cutoff distance for the GNM. Default is 7.3 A.
                
                Returns:
                --------
                    HBar: double
                        The Hessian matrix of the system considering the environment.
            %}
            Hss = -getContactMatrix(system, system, cutOff);
            Hee = -getContactMatrix(environment, environment, cutOff);
            Hse = -getContactMatrix(environment, system, cutOff);
            
            ssDiag = sum(Hss, 2) + sum(Hse, 2);
            eeDiag = sum(Hee, 1) + sum(Hse, 1);
            Hss  = Hss - diag(ssDiag);
            Hee  = Hee - diag(eeDiag);
            [eeIvec, eeIval] = eig(Hee);
            eigZero = diag(eeIval) < 10^-10;
            eeIval = eeIval^-1;
            for i = find(eigZero)
                eeIval(i, i) = 0;
            end
            ee_inv = eeIvec * eeIval * eeIvec';
            obj.HBar = Hss - Hse * ee_inv * Hse';
        end

        function obj = get(obj, HBar)
            %{
                Computes the GNM modes considering the environment.
                Parameters:
                -----------
                    system: struct
                        The system can be a single chain or a set of chains, a protein or a complex.
                        The system coordinates are retrieved by system.coord
                
                    environment: struct
                        The environment can be a single chain or a set of chains, a protein or a complex.
                        The environment coordinates are retrieved by environment.coord
                    
                    cutOff: double, optional
                        The cutoff distance for the GNM. Default is 7.3 A.
                
                Returns:
                --------
                    system: struct
                        The system with the environment GNM modes in the field system.envGNM
                
                    GVals: double
                        The eigenvalues of the environment GNM modes 
            %} 
            [GVecs, GVals]  = eig(HBar);
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

        function [GVecs, GVals] = getModes(temp_H_bar, save_dir)
            %{
                Computes the GVecs, GVals as loading temp_H_bar.mat file.
                Parameters:
                -----------
                    temp_H_bar: mat file
                        The temp_H_bar matrix file. temp_H_bar.mat
                    save_dir: string
                        The directory to save the GVecs, GVals mat files.
                        temp_U_all.mat: GVecs, N x (N-1)
                        temp_S_inv_all.mat: 1/GVals, N-1
                
                Returns:
                --------
                    GVecs: double, N x (N-1)
                        The eigenvectors of the environment GNM modes 
                    GVals: double, N-1
                        The eigenvalues of the environment GNM modes
            %}
            load(temp_H_bar); % H_bar matrix
            [U, S] = eig(H_bar);
            S = diag(S);
            [S, ind] = sort(S, 'ascend');
            U = U(:, ind);
        
            if S(1) > 10^(-9)
                error('Useful_func:eigError','warning: the first eigenvalue is not zero.');
            end
            if S(2) < 10^(-9)
                error('Useful_func:eigError','warning: There are more than one zero eigenvalue.')
            end
            
            GVals = S(2:end);
            GVecs = U(:, 2:end);
            
            S = 1 ./ S(2:end);
            U = U(:, 2:end);
        
            % If save_dir is provided, save the data
            if exist('save_dir', 'var')
                U_all_mat = fullfile(save_dir, 'temp_U_all.mat');
                S_inv_all_mat = fullfile(save_dir, 'temp_S_inv_all.mat');
                save(U_all_mat, 'U', '-v7.3');
                save(S_inv_all_mat, 'S', '-v7.3');
                % Print the files that are saved
                fprintf('The following files are saved: temp_U_all.mat, temp_S_inv_all.mat\n');
            end
        end 

        function [H_bar] = getHbar(kdat)
            %{
                Computes the H_bar matrix as loading kdat file.
                Parameters:
                -----------
                    kdat: mat file
                        The kdat matrix file. temp.kdat
                
                Returns:
                --------
                    H_bar: double, N x N
                        The H_bar matrix of the system considering the environment.
            %}
            [I,J,K] = textread(kdat, '%f%f%f');
            IJK=[I,J,K];
            ind_sys=find(J <= SN);
            IJK_s=IJK(ind_sys,:);
            ind_env=find(J > SN);
            IJK_e=IJK(ind_env,:);
        
            IJK_s=[IJK_s(:,1) IJK_s(:,2) IJK_s(:,3);IJK_s(:,2) IJK_s(:,1) IJK_s(:,3)];
            Hss=sparse(IJK_s(:,1),IJK_s(:,2),IJK_s(:,3),SN,SN);
            Hss = spdiags(1/2*spdiags(Hss,0),0,Hss); % Any elements of s which have duplicate values of i and j are added together.
            
            ind_es=find(IJK_e(:,1) <= SN);
            IJKse=IJK_e(ind_es,:);
            [r,c]=size(IJKse); % r is the number of non-zero elements in the Hse
        
            Hse=sparse(IJKse(:,1),IJKse(:,2)-SN*ones(r,1),IJKse(:,3),SN,resnum-SN);
            
            ind_ee=find(IJK_e(:,1) > SN);
            IJKee=IJK_e(ind_ee,:);
            [r1,c1]=size(IJKee); % r1 is the number of non-zero elements in the Hee
            IJKee=[IJKee(:,1)-SN*ones(r1,1) IJKee(:,2)-SN*ones(r1,1) IJKee(:,3);IJKee(:,2)-SN*ones(r1,1) IJKee(:,1)-SN*ones(r1,1) IJKee(:,3)];
            Hee=sparse(IJKee(:,1),IJKee(:,2),IJKee(:,3),resnum-SN,resnum-SN);
            Hee = spdiags(1/2*spdiags(Hee,0),0,Hee);
            
            Hee=full(Hee);
            [Uee,See]=eig(Hee);
            See1=diag(See);
        
            num_zero_eig=length(find(See1<10^-12));
            num_zero_eig_Hee=num_zero_eig
            See1=1./See1(num_zero_eig+1:end);
            See=diag(See1);
            Hee_inv=Uee(:,(num_zero_eig+1):(resnum-SN)) * See * Uee(:,(num_zero_eig+1):(resnum-SN))';
            
            % H_bar = Hss - Hse Hee-1 HseT
            H_bar = full(Hss) - full(Hse)*Hee_inv*full(Hse)' ;
        end
    end
end