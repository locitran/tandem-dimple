classdef getFeatures
    %{
        Class for computing features from PDB file.
        
        Properties:
        -----------
        ENM: char
        Elastic Network Model (ENM), either 'GNM' or 'envGNM'.
        
        protein: struct
        Protein structure, containing amino acid information.
        
        system: struct
        System structure, containing CA atoms only
        
        environment: struct
        Environment structure, containing membrane and nucleic atoms (if any), then, envGNM is used.
        The used names (system and environment) are based on 
        DynOmics paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5793847/

        n_residues: double
        Number of residues in the protein
        
        sequence: char
        Amino acid sequence of the protein, separated by chain
        
        GVecs: double
        Eigenvectors of the ENM
        
        GVals: double
        Eigenvalues of the ENM
                
        Example:
        --------
        addpath('./matFeatures');
        importlib;
        pdbPath = '1G0D.pdb';
        data = getFeatures(pdbPath);
        data = data.get_GVecs_GVals();
        data = data.protein_level_features(data.GVecs, data.GVals);
        data = data.residue_level_features();
        data = data.removeField();

        All features: 'entropy_v\trmsf_overall\teig_first\teig_sec\tSEG_all\tSEG_20\teig5_eig1\trank_1\trank_2\tvector_1\tvector_2\tGNM_co\tco_rank\teig_vv_1\teig_vv_2\tca_len\tgyradius\tRPA_1\tRPA_2\tRPA_3\tloop_percent\thelix_percent\tsheet_percent\tside_chain_length\tIDRs\tdssp_result\tDcom\tphilic_percent\tphobic_percent\tcontact_per_res\tpolarity\tcharge\tpka_charge\tsasa\tssbond_matrix\th_bond_group\tatomic_1\tatomic_3\tatomic_5\tconsurf\tACNR\tdelta_gyradius\tdelta_side_chain_length\tdelta_sasa\tdelta_phobic_percent\tdelta_philic_percent\tdelta_contact_per_res\tdelta_polarity\tdelta_charge\tdelta_pka_charge\tdelta_ssbond_matrix\tdelta_h_bond_group\twt_PSIC\tDelta_PSIC\tBLOSUM\tentropy\tranked_MI\tANM_effectiveness-chain\tANM_sensitivity-chain\tstiffness-chain';
        33 TANDEM features: 'atomic_1\tvector_2\tco_rank\tatomic_3\tatomic_5\tDcom\tvector_1\trank_2\teig_first\tphobic_percent\teig_sec\tgyradius\tside_chain_length\trank_1\trmsf_overall';
    %}
    properties
        ENM             {char}
        protein         {struct}
        system          {struct}
        environment     {struct}
        n_residues      {double}
        sequence        {char}
        GVecs           {double} 
        GVals           {double}        
    end

    %~ Constructor method
    methods
        
        function obj = getFeatures(pdbPath)
            data            = getFeatures.parsePDB(pdbPath);
            obj.sequence    = data.sequence;
            obj.n_residues  = data.n_residues;
            obj.protein     = data.protein;
            
            % Construct system and environment
            obj.system      = data.ca;
            obj.environment = [data.membrane, data.na_CG];

            if ~isempty(obj.environment)
                obj.ENM = 'envGNM';
            else
                obj.ENM = 'GNM';
            end
        end

        function obj = get_GVecs_GVals(obj, cutOff)
            %{
                Compute modes from structure.
                Parameters:
                -----------
                cutOff : double
                    Cutoff distance for the ENM.
            %}
            if ~exist('cutOff', 'var')
                cutOff = 7.3;
            end
            [GVecs, GVals] = getFeatures.getModes(obj.system, obj.environment, obj.ENM, cutOff);
            [obj.GVecs, obj.GVals] = deal(GVecs, GVals);
        end

        %~ Get overall protein features
        function obj = protein_level_features(obj, GVecs, GVals)
            %{
                Compute dynamic features.
                Parameters:
                -----------
                GVecs : double
                    Eigenvectors of the ENM.
                GVals : double
                    Eigenvalues of the ENM.
            %}
            N = obj.n_residues;
            if length(GVals) ~= N - 1
                error('length(GVals) ≠ N - 1', 'length(GVals) = %d, N = %d', length(GVals), N);
            end
            if size(GVecs) ~= [N, N-1]
                error('size(GVecs) ≠ N x (N-1)', 'size(GVecs) = %d x %d, N = %d', size(GVecs), N);
            end
            
            %~ Set dynamic features
            % entropy_v                  = (0.5*N) + (((N * log(2*pi)) - sum(log(GVals))) * 0.5);
            rmsf_overall               = sqrt(sum(1./GVals) / N);
            [eig_first, eig_sec]        = deal(GVals(1), GVals(2));
            % [SEG_all, SEG_20]          = getFeatures.getShannonEntropy(GVals, N); % Must check
            % eig5_eig1                  = log10(GVals(5)) - log10(GVals(1));
            rank_1                      = getFeatures.getRank_EVec(GVecs(:, 1), N);
            rank_2                      = getFeatures.getRank_EVec(GVecs(:, 2), N);
            [vector_1, vector_2]        = deal(abs(GVecs(:, 1)), abs(GVecs(:, 2)));
            % eig_vv_1                   = 1./GVals(1) * abs(GVecs(:, 1));
            % eig_vv_2                   = 1./GVals(2) * abs(GVecs(:, 2));
            [displacement, co_rank]     = getFeatures.getCov_matrix(GVecs, GVals);
            %~ End dynamic features
            gyradius                    = getFeatures.getRadiusOfgyration(obj.protein); % Structure feature
            % [RPA1, RPA2, RPA3]         = getFeatures.getPCAfeature(obj.system);% Structure feature
            
            for i = 1:numel(obj.system)
                %~ Set dynamic features
                % obj.system(i).entropy_v        = entropy_v;
                obj.system(i).rmsf_overall     = rmsf_overall;
                obj.system(i).eig_first         = GVals(1);
                obj.system(i).eig_sec           = GVals(2);
                % obj.system(i).SEG_all          = SEG_all;
                % obj.system(i).SEG_20           = SEG_20;
                % obj.system(i).eig5_eig1        = log10(GVals(5)) - log10(GVals(1));;
                obj.system(i).rank_1            = rank_1(i);
                obj.system(i).rank_2            = rank_2(i);
                obj.system(i).vector_1          = vector_1(i);
                obj.system(i).vector_2          = vector_2(i);
                obj.system(i).GNM_co            = displacement(i);
                obj.system(i).co_rank           = co_rank(i);
                % obj.system(i).eig_vv_1         = eig_vv_1(i);
                % obj.system(i).eig_vv_2         = eig_vv_2(i);
                %~ Set structure features
                % obj.system(i).ca_len           = N;
                obj.system(i).gyradius          = gyradius;
                % obj.system(i).RPA_1             = RPA1;
                % obj.system(i).RPA_2             = RPA2;
                % obj.system(i).RPA_3             = RPA3;
                %~ Set chemical features
            end
        end

        function obj = residue_level_features(obj)
            % IDRs                = cellfun(@(x) getFeatures.getIDRs(x), obj.sequence, 'uniformoutput', 0);
            % IDRs                = cat(1,IDRs{:});
            % for i = 1:numel(obj.system)
            %     obj.system(i).IDRs              = IDRs(i);                  % Structure feature
            % end

            side_chain_length   = getFeatures.getSideChainLength(obj.system);
            Dcom                = getFeatures.getDcom(obj.protein);
            [atomic]                = getFeatures.getAtomicContactPerRes(obj.protein, 4.5, [0, 1, 2], [1, 2, 3]);
            % contact_per_res         = mean(atomic(:, 1));
            polarity                = getFeatures.getPolarity(obj.system);
            charge                  = getFeatures.getCharge(obj.system);    % logical
            charge                  = double(charge);                       % double
            phobic_percent              = getFeatures.getPhobicPercent(obj.system); % Chemical feature
            % philic_percent             = 100 - phobic_percent;                     % Chemical feature

            for i = 1:numel(obj.system)
                obj.system(i).side_chain_length = side_chain_length(i);     % Structure feature
                obj.system(i).Dcom              = Dcom(i);                  % Structure feature
                % obj.system(i).contact_per_res   = contact_per_res;
                obj.system(i).polarity          = polarity(i);
                % obj.system(i).charge            = charge(i);
                % obj.system(i).ssbond_matrix     = isSSBond(i);
                obj.system(i).atomic_1          = atomic(i, 1);
                obj.system(i).atomic_3          = atomic(i, 2);
                obj.system(i).atomic_5          = atomic(i, 3);
                obj.system(i).phobic_percent    = phobic_percent;
                % obj.system(i).philic_percent    = philic_percent;
            end
        end

        %~ Remove field from ca
        function obj = removeField(obj)
            obj.system = rmfield(obj.system, {'record', 'atomno', 'atmname', 'coord', 'bval', 'occupancy', 'alternate', 'segment', 'elementSymbol', 'internalResno'});
        end
    end

    methods (Static)
        
        function [consurf_score]      = get_consurf(tsv_file, resno_resname)
            %{
                Parse ConSurf file.
                Parameters:
                -----------
                tsv_file : char
                    Path to ConSurf file.
                resno_resname : cell
                    Residue number and name. E.g., 'GLY6'.

                Returns:
                --------
                consurf_data : struct
                    ConSurf data.
            %}
            tsv_data = readtable(tsv_file, 'FileType', 'text', 'Delimiter', '\t', 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');
            consurf_score = nan(size(resno_resname));
            [is_member, loc] = ismember(resno_resname, tsv_data.ATOM);
            consurf_score(is_member) = tsv_data.SCORE(loc(is_member));
        end

        function [GVecs, GVals]      = getModes(system, environment, ENM, cutOff)
            %{
                Compute modes from structure or from temp_U_all.mat and temp_S_inv_all.mat files.
                Parameters:
                -----------
                cutOff : double
                    Cutoff distance for the ENM.

                Returns:
                --------
                GVecs : double
                    Eigenvectors of the ENM.
                GVals : double
                    Eigenvalues of the ENM.
            %}
            if ~exist('ENM', 'var')                 % Check if ENM is provided
                error('ENM parameter is missing. Please provide either "GNM" or "envGNM".');
            end
            validENMValues = {'GNM', 'envGNM'};     % Check if ENM is valid
            if ~ismember(ENM, validENMValues)
                error('Invalid value for ENM parameter. Please use either "GNM" or "envGNM".');
            end
            if ~exist('cutOff', 'var')
                cutOff = 7.3;
            end

            if strcmp(ENM, 'envGNM')
                data = envGNM(system, environment, cutOff);
            elseif strcmp(ENM, 'GNM')
                data = GNM(system, cutOff);
            end
            [GVecs, GVals] = deal(data.GVecs, data.GVals);
        end

        function [data]              = parsePDB(pdbPath, consurf)
            %{
                Parse information in PDB file.
                Parameters:
                -----------
                pdbPath : char
                    Path to PDB file.
                consurf : bool, default = false
                    If True, parse pdbPath with removing the chain which has <30 residues.
                    In ConSurf, there must be at least 20 residues in the ATOM records, 
                    or else the job is rejected. (https://consurf.tau.ac.il/consurf_quick_help.php)

                Returns:
                --------
                data : object
                    data with full, protein, ca, membrane, na, HETATM, n_residues, and sequence attributes.

                Example:
                --------
                pdbPath = '1aqd.pdb'
                data = getFeatures.parsePDB(pdbPath, true)
            %}
            full        = readPDB(pdbPath);
            protein     = as('protein', full);
            condition   = strcmp({protein.record}, 'ATOM');
            protein     = protein(condition);
            ca          = as('name CA', protein);
            membrane    = as('resn NE1 and name Q1', full);
            na          = as("nucleic", full);
            na_CG       = as("name P C4' C2", na);
            HETATM      = as('HETATM', full);
            ligand      = struct();

            % Create struct for ligand
            ligand_resn = unique({HETATM.resname});                     % Array of ligand resnames
            for j = 1:length(ligand_resn)                           
                ligand_j            = ligand_resn{j};                   % Component of ligand_resn
                ligand_j_fieldname  = ['ligand_' ligand_j];             % ligand_j_fieldname = 'ligand_name'
                ligand_j_struct     = as(['resn ' ligand_j], HETATM);
                ligand.(ligand_j_fieldname) = ligand_j_struct;          % ligand{model}.ligand_resn = ligand_j_struct
            end
            
            sequence     = getSequence(ca); % dtype: cell
            seq_lengths = zeros(1, numel(sequence));
            for i = 1:numel(sequence)
                seq_lengths(i) = length(sequence{i});
            end

            % Consider consurf parser
            if exist('consurf', 'var') && consurf
                cumulative_lengths = cumsum(seq_lengths); % Calculate cumulative sum of sequence lengths
                indices_to_remove = []; % Initialize indices to remove
                % Iterate over cumulative lengths to find chains with <20 residues
                for i = 1:length(cumulative_lengths)
                    start_index = 1;
                    if i > 1
                        start_index = cumulative_lengths(i-1) + 1;
                    end
                    end_index = cumulative_lengths(i);
                    % Check if chain has <20 residues
                    if (end_index - start_index + 1) < 20
                        % Add indices of residues in the chain to indices_to_remove
                        indices_to_remove = [indices_to_remove, start_index:end_index];
                    end
                end
                % Remove chains with <20 residues from ca struct
                ca(indices_to_remove) = [];

                % Modify sequence and seq_lengths to reflect changes
                sequence = getSequence(ca);
                seq_lengths = zeros(1, numel(sequence));
                for i = 1:numel(sequence)
                    seq_lengths(i) = length(sequence{i});
                end
            end

            % Reset ca.internalResno starting from 1
            for i = 1:length(ca)
                ca(i).internalResno = i;
            end

            % Print out description of PDB file (continued)
            n_residues   = length(ca);
            ligand_resn = fieldnames(ligand);
            ligand_resn = strjoin(ligand_resn, ', ');
            fprintf('> %d chain(s), %d protein atoms, %d residues\n', length(sequence), length(protein), n_residues);
            fprintf('> %d membrane atoms (NE1), %d nucleic atoms, %d HETATM atoms (%s)\n', length(membrane), length(na), length(HETATM), ligand_resn);

            % Assign attributes to data
            data.full = full;
            data.protein = protein;
            data.ca = ca;
            data.membrane = membrane;
            data.na = na;
            data.na_CG = na_CG;
            data.HETATM = HETATM;
            data.ligand = ligand;
            data.n_residues = n_residues;
            data.sequence = sequence;
        end

        function [IDRs]              =  getIDRs(sequence)
            [pathstr,~,~] = fileparts(mfilename('fullpath'));
            tempDir = fullfile(pathstr, 'temp');
            template = '>%s\n%s\n';
            fastaSeq = sprintf(template, 'TMP', sequence);
            fileName = [tempDir char(java.util.UUID.randomUUID)];
            tmpfile = fopen(fileName,'w');
            fprintf(tmpfile,'%s',fastaSeq);
            try
                [exitCode, data] = system(['RONN ' fileName]);
                if exitCode ~= 0
                    error('Useful_func:IntrinsicDisorder','Failed to call RONN.');
                end
                IDRs =  sscanf(data(6:end),'%f');
            catch e
                fclose(tmpfile);
                delete(fileName);
                rethrow(e);
            end
            fclose(tmpfile);
            delete(fileName);
        end     

        function [polarity]          = getPolarity(pdbStructureOrSequence)
            order = 'LPMWAVFIGSTCNQYHDEKR';
            % res_name = {'LEU','PRO','MET','TRP','ALA','VAL','PHE','ILE','GLY','SER','THR','CYS','ASN','GLN','TYR','HIS','ASP','GLU','LYS','ARG'};
            polaritys = [4.9 8 5.7 5.4 8.1 5.9 5.2 5.2 9 9.2 8.6 5.5 11.6 10.5 6.2 10.4 13 12.3 11.3 10.5];
            
            if isstruct(pdbStructureOrSequence)
                allAtomNames = {pdbStructureOrSequence.atmname};
                noh = pdbStructureOrSequence(cellfun(@isempty, regexp(allAtomNames, 'H*')));
                chainSeq = getSequence(noh);
                seq = [];
                for i = 1: length(chainSeq)
                    seq = [seq chainSeq{i}];
                end
            elseif ischar(pdbStructureOrSequence)
                seq = pdbStructureOrSequence;
            else
                error('UsefulFunc:Polarity', 'Input should PDB structure or sequence (char)')
            end
            
            [~,index] = ismember(seq, order);
            polarity = polaritys(index);
        end

        function [ssbond_feature, protein_hole_ss_bond] = getssbondMatrix(protein)
            pdb_CYS = as('resn CYS', protein);
            ca = as('name CA', protein);        
            ss = as('name SG', pdb_CYS);
            [ss_contact] = getContactMatrix(ss, ss, 2.4);
            ss_contact(eye(size(ss_contact)) ~= 0) = 0;
            
            %real_contact = ss_contact & angle_matrix;
            real_contact = ss_contact;
            ss_interno = [ss.internalResno];
            close_matrix = abs(bsxfun(@minus, ss_interno', ss_interno));
            close_matrix(close_matrix == 1) = 0;
            close_matrix(close_matrix > 1) = 1;
            
            %find neighbor atom
            no_close_matrix = real_contact & close_matrix;
            %remove neighbor atom in real_contact matrix
            
            ssDistance = getPairwiseDistance(ss, ss);
            ssDistance(ssDistance >2.4) = 0;
            real_ssDistance = ssDistance.*no_close_matrix;
            real_ssDistance(real_ssDistance == 0) = 3;

            remove_double_contact = zeros(length(real_ssDistance), length(real_ssDistance));
            for i = 1: length(real_ssDistance)
                one_matrix = real_ssDistance(i, 1: length(real_ssDistance));
                close_num = min(one_matrix);
                if close_num == 3
                    index = one_matrix == close_num;
                    remove_double_contact(i, index) = 0;
                else
                    index = find(one_matrix == close_num);
                    remove_double_contact(i, index) = close_num;
                end
            end
            result_contact_matrix = remove_double_contact & real_contact;
            contact_index = find(sum(result_contact_matrix) == 1);
            contact_internalResno = [ss(contact_index).internalResno];

            ca_internal_matrix = [ca.internalResno];
            ssbond_feature = zeros(length(ca), 1);
            for i = 1: length(contact_index)
                contact_resno = contact_internalResno(i);
                contact_inca_index = find(ca_internal_matrix == contact_resno);
                ssbond_feature(contact_inca_index,1) = 1;
            end
            count_ssbond = sum(ssbond_feature);
            protein_hole_ss_bond = count_ssbond / 2;
        end

        function [RPA1, RPA2, RPA3]  = getPCAfeature(ca)
            ca_center = getGeometrycenter(ca);
            [vecs, vals] = getPrincipalAxis(ca);
            vals = diag(vals);
            RPA1 = sqrt(vals(3) / vals(1));
            RPA2 = sqrt(vals(2) / vals(1));
            RPA3 = sqrt(vals(3) / vals(2));
        end            

        function [stand_dist,res_dist,res_center] = getDcom(protein)
            %{
                Compute the distance between the center of mass (COM) of the protein and the COM of each residue.
                Parameters:
                -----------
                protein : struct
                    Protein structure.
                ca : struct
                    CA atoms of the protein structure.

                Returns:
                --------
                stand_dist : double
                    Standardized distance between the COM of the protein and the COM of each residue.
                res_dist : double
                    Distance between the COM of the protein and the COM of each residue.
                res_center : double
                    COM of each residue.
            %}
            protein = assignMass(protein);
            protein_center = getCenterOfMass(protein);
            [unique_elements, ~, idx] = unique([protein.internalResno]);
            count_internal = histcounts(idx, 1:(length(unique_elements) + 1));
            count_internal(count_internal == 0) = '';
            res_center = zeros(1, 1);
            n = 0;
            m = 1;
            for i = 1: length(count_internal)
                n = n + count_internal(i);
                res_cen = getCenterOfMass(protein(m: n));
                res_center(i, 1) = res_cen(1);
                res_center(i, 2) = res_cen(2);
                res_center(i, 3) = res_cen(3);
                m = m +count_internal(i);
            end
            res_dist = zeros(1, 1);
            res_dist = sqrt(sum((res_center - protein_center) .^ 2, 2));
            stand_dist = (res_dist - mean(res_dist)) ./ std(res_dist);
        end  

        function [gyradius]          = getRadiusOfgyration(protein)
            num_atom_list = getAtomNumPerRes(protein);
            res_num = length(num_atom_list);
            end_indeces = cumsum(num_atom_list);
            start_indeces = zeros(1, length(end_indeces));
            start_indeces(2: end) = end_indeces(1: end-1);
            start_indeces = start_indeces + 1;
            res_centers = cell2mat(arrayfun(@(start_index, end_index)getGeometrycenter(protein(start_index: end_index)), start_indeces, end_indeces, 'UniformOutput', 0)');
            res_center = mean(res_centers);
            gyradius = sqrt(sum(sum( (res_centers - repmat(res_center, size(res_centers, 1), 1)) .^ 2 )) / res_num);
        end

        function [sideChainLength]   = getSideChainLength(pdbStructureOrSequence)
            order = 'GAVLIMFWPSTCYNQDEKRH';
            numSideChainHeavyAtom = [0,1,3,4,4,4,7,10,3,2,3,2,8,4,5,4,5,5,7,6];
            
            if isstruct(pdbStructureOrSequence)
                allAtomNames = {pdbStructureOrSequence.atmname};
                noh = pdbStructureOrSequence(cellfun(@isempty, regexp(allAtomNames, 'H*')));
                chainSeq = getSequence(noh);
                seq = [];
                for i = 1: length(chainSeq)
                    seq = [seq chainSeq{i}];
                end
            elseif ischar(pdbStructureOrSequence)
                seq = pdbStructureOrSequence;
            else
                error('UsefulFunc:SideChainLength','Input should PDB structure or sequence (char)')
            end
            
            [~,index] = ismember(seq,order);
            sideChainLength = numSideChainHeavyAtom(index);
        end            

        function [res_contact]       = getContact(ca, cutoff, res_list)
            %{
                Find contacts for each residue in a protein in residue level.
                Parameters:
                -----------
                protein : struct
                    Protein structure.
                cutoff : double, default = 7.3 angstrong
                    Cutoff distance for the contact matrix.
                res_list : cell
                    List of residues to compute contact for.
                    The format of each residue is res_identifier
                    res_identifier := {chainID}-{resName}{resID}{iCode}. E.g., 1G0D.pdb: A-GLY6, A-LEU7, A-ILE8, etc.

                Returns:
                --------
                res_contact : cell
                    Contact list for each residue.
            %}
            if ~exist('cutoff', 'var')
                cutoff = 7.3;
            end
            n_residues = length(ca);
            contactM = getContactMatrix(ca, ca, cutoff);
            % Set 0 for diagonal elements
            contactM(eye(size(contactM)) ~= 0) = 0;

            % if resID not provided, return res_contact for all residues
            % if resID provided, return res_contact for the specified residues
            if exist('res_list', 'var')
                res_contact = cell(length(res_list), 1);
                
                for i = 1:length(res_list)
                    res_identifier = res_list{i};
                    chainID = res_identifier(1);
                    resName = res_identifier(3:5);
                    % Check iCode: if res_identifier(-1) is not a number, then iCode = res_identifier(-1)
                    if ~isstrprop(res_identifier(end), 'digit')
                        resID = res_identifier(6: end-1);
                        iCode = res_identifier(end);
                    else
                        resID = res_identifier(6: end);
                        iCode = ' ';
                    end

                    % Find target residue 
                    residue = ca(...
                        strcmp({ca.subunit}, chainID)  & ...   % Match chainID
                        strcmp({ca.resno}, resID)      & ...   % Match resno
                        strcmp({ca.resname}, resName)  & ...   % Match resName
                        strcmp({ca.iCode}, iCode)      & ...   % Match iCode
                        strcmp({ca.atmname}, 'CA')       ...   % Match CA atoms
                    );

                    % If residue not found, return empty cell
                    if isempty(residue)
                        res_contact{i} = {};
                    else
                        % Extract the contact list for the target residue
                        contact_indices = find(contactM(residue.internalResno, :));
                        res_contact{i} = cell(1, length(contact_indices));

                        % Convert contact_indices into {chainID}-{resName}{resID}{iCode}
                        for j = 1:length(contact_indices)
                            res_ = ca(...
                                [ca.internalResno] == contact_indices(j) & ...
                                strcmp({ca.atmname}, 'CA') ...
                            );
                            res_contact{i}{j} = [res_.subunit '-' res_.resname num2str(res_.resno) res_.iCode];
                            res_contact{i}{j} = strtrim(res_contact{i}{j});  % Remove whitespace and update
                        end             
                    end          
                end
            else % Extract the contact list for each residue
                
                res_contact = cell(n_residues, 1);
                for i = 1:n_residues
                    contact_indices = find(contactM(i, :)); % res_index in contact with
                    % res_contact{i} = cell(1, length(contact_indices)); % Preallocate res_contact{i}
                    res_contact{i} = [];
                    % Convert res_index into {chainID}-{resName}{resID}{iCode}
                    for j = 1:length(contact_indices)
                        res_ = ca(...
                        [ca.internalResno] == contact_indices(j) & ...
                        strcmp({ca.atmname}, 'CA') ...
                        );
                        % res_contact{i}{j} = [res_.subunit '-' res_.resname num2str(res_.resno) res_.iCode];
                        % res_contact{i}{j} = strtrim(res_contact{i}{j}); % Remove whitespace and update
                        res_contact{i} = [res_contact{i}, res_.consurf];
                    end
                    % Average the consurf score of the contact residues, if containing NaN, result is NaN
                    res_contact{i} = mean(res_contact{i}, 'omitnan');
                end
            end
        end

        function [atomic]            = getAtomicContactPerRes(protein, cutoff, groupNeighbor, skipNeighbor)
            n_atom = length(protein);
            n_atom_per_res = getAtomNumPerRes(protein);
            n_residue = length(n_atom_per_res);
            contactM = getContactMatrix(protein, protein, cutoff);
            index = zeros(n_residue + 1,1 ); % Initialize array of zeros, (N+1) X 1
            index(2:end) = cumsum(n_atom_per_res); % Cumulative sum of n_atom_per_res
            n_res_per_group = zeros(1, n_residue);
            groupByGroup = zeros(n_atom, n_residue);
    
            groupNeighbor = [0, 1, 2];
            skipNeighbor  = [1, 2, 3];
            atomic = zeros(n_residue, 3);
    
            for idx = 1:numel(groupNeighbor)
                group = groupNeighbor(idx);
                skip = skipNeighbor(idx);

                for i = 1:n_residue
                    startResIndex = i - group;
                    endResIndex = i + group;
                    if startResIndex <1
                        startResIndex = 1;
                    end
                    if endResIndex > n_residue
                        endResIndex = n_residue;
                    end
                    n_res_per_group(i) = endResIndex - startResIndex + 1;
                    groupByGroup(:,i) = sum(contactM(:, index(startResIndex) + 1 : index(endResIndex+1)), 2);
                end
                
                groupByGroup = groupByGroup > 0;
                for i = 1:n_residue
                        startResIndex = i - group - skip; 
                        endResIndex = i + group + skip;
                    if startResIndex < 1
                        startResIndex = 1;
                    end
                    if endResIndex > n_residue
                        endResIndex = n_residue;
                    end
                    groupByGroup(index(startResIndex) + 1 : index(endResIndex+1), i) = 0;
                end
                groupTotalContact = sum(groupByGroup, 1) ./ n_res_per_group;

                % add groupTotalContact vector to atomic
                atomic(:, idx) = groupTotalContact';
                % How to get the first column of atomic

            end
        end

        function [ca_charge]         = getCharge(ca)
            ca = as('name CA', ca);
            ca_charge = as('resn ASP GLU HIS LYS ARG', ca, 1);
        end

        function [phobic_precent]    = getPhobicPercent(ca)
            N = length(ca);
            phobic_matrix = as('resn GLY ALA VAL LEU ILE MET PHE TRP PRO', ca, 1)';
            phobic_precent = (sum(phobic_matrix) / N)*100;
        end

        function [displacement, rank] = getCov_matrix(GVecs, GVals)
            N = length(GVals) + 1;
            cov_matrix = GVecs * diag(1./GVals) * GVecs';
            displacement = diag(cov_matrix);
            displacement = abs(displacement);
            [~, ~, rank] = unique(displacement);
            rank = max(rank) - rank + 1;
            rank = 1 - (rank ./ N);
        end

        function [rank_EVec]         = getRank_EVec(EVec, n_residues)
            EVec = abs(EVec);
            [~, ~, rank_EVec] = unique(EVec);
            rank_EVec = max(rank_EVec) - rank_EVec + 1;
            rank_EVec = 1 - (rank_EVec ./ n_residues);
        end

        function [SEG_all, SEG_20]   = getShannonEntropy(GVals, n_residues)
            binrange = linspace(GVals(1), GVals(end), n_residues);
            bincounts = histc(GVals, [binrange]);
            bincounts = bincounts(1: end-1);
            bincounts(end) = bincounts(end) + 1;
            bincounts(end + 1) = 0;
            bincounts = bincounts(bincounts ~= 0);
            len_ca = sum(bincounts);
            
            SEG_all = ((-1) * sum((bincounts ./ len_ca) .* log2(bincounts ./ len_ca))) / log2(length(GVals));
        
            top_20 = round(length(GVals) * 0.2);
            top_20_GVals = GVals(1: top_20);
            binrange = linspace(top_20_GVals(1), top_20_GVals(end), length(top_20_GVals) + 1);
            bincounts = histc(top_20_GVals, [binrange]);
            bincounts = bincounts(1: end-1);
            bincounts(end) = bincounts(end) + 1;
            bincounts(end + 1) = 0;
            bincounts = bincounts(bincounts ~= 0);
            len_ca = sum(bincounts);

            SEG_20 = ((-1) * sum((bincounts ./ len_ca) .* log2(bincounts ./ len_ca))) / log2(length(top_20_GVals));
        end
    end
end

