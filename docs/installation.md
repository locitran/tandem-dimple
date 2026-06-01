# Docker

**Step 1**: Assume you are in `test` folder :  `./test/`
clone the folder
```bash
mkdir tandem
cd tandem
git clone https://github.com/locitran/tandem-dimple.git .
```

Now we should be in the path/to/tandem folder.
`ls` command should show `src/` `data/` `models/` etc.

**Step 2**: Build docker image using original Dockerfile (please not put databases inside docker images)
```bash
docker build -t tandem -f docker/Dockerfile .
```

**Step 3**: Download 2 databases: 
```bash
bash scripts/download_pfam.sh data/pfamdb # 1.5G, ~1.5m
bash scripts/download_consurf_db.sh data/consurf/db # 2.5G, ~2m
# Please skip this database for now
# We will download this database later
bash scripts/download_uniref90.sh data/consurf # 90G, ~127m
# Consider downloading uniref50 instead of uniref90 due to the size
bash scripts/download_uniref50.sh data/consurf # 26G, ~60m
```
After this step, we will have `path/to/tandem//data/pfamdb` folder and `path/to/tandem/data/consurf/db/2024-10-08` folder. Make sure you have the correct path to these databases.

**Step 4**: Run image/container
```bash
# cwd: path/to/tandem
# pfamdb: path/to/tandem/data/pfamdb
# consurf: path/to/tandem/data/consurf
docker run -it \
  -v .:/tandem \
  -w /tandem \
  tandem:latest bash \
  -c "source activate tandem && python test/input_as_list_SAVs.py"
```

# Build from Scratch, without Docker

**Step 1**: Assume you are in `test` folder :  `./test/`
clone the folder
```bash
mkdir tandem
cd tandem
git clone https://github.com/locitran/tandem-dimple.git .
```

**Step 2**: Install dependencies
We assume you have already installed Anaconda or Miniconda. If not, please install it first.
```bash
conda create -n tandem python=3.11.11
conda activate tandem
pip install -r requirements.txt
sudo apt install hmmer
```

**Step 3**: Download 2 databases: 
```bash
bash scripts/download_pfam.sh data/pfamdb # 1.5G, ~1.5m
bash scripts/download_consurf_db.sh data/consurf/db # 2.5G, ~2m
# Please skip this database for now
# We will download this database later
bash scripts/download_uniref90.sh data/consurf # 90G, ~127m
bash scripts/download_uniref50.sh data/consurf # 

# dependencies for ConSurf tool
conda install -c conda-forge -c bioconda mmseqs2
sudo apt install cd-hit
sudo apt install prottest 
sudo apt install ncbi-blast+
sudo apt install mafft
sudo apt install clustalw
sudo apt install muscle
sudo apt install hmmer
```
**Step 4**: Test the installation
```bash
python test/input_as_list_SAVs.py
```

# INSTALLING PolyPhen-2 STANDALONE SOFTWARE
Reference: https://genetics.bwh.harvard.edu/wiki/!pph2/_media/hg0720.pdf or [pph2-documentation.pdf](pph2-documentation.pdf).

```bash
################# Download Source Code #################
wget 'https://genetics.bwh.harvard.edu/wiki/!pph2/_media/polyphen-2.2.3r408.tar.gz'
tar -xvzf polyphen-2.2.3r408.tar.gz
```



apt-get update && apt-get install -y build-essential curl wget
apt update --quiet \
    && apt install --yes --quiet build-essential curl wget \ 
    && apt install --yes --quiet libdb-dev libxml2-dev libxslt1-dev zlib1g-dev libssl-dev \
    && apt install --yes --quiet libxml-simple-perl libwww-perl libdbd-sqlite3-perl \
    && apt install --yes --quiet libcgi-pm-perl bioperl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

```bash
docker run -dit --name pph2 \
  -v /home/loci/main/tandem_website_dev/tandem/PolyPhen-2:/pph2 \
  ubuntu:20.04 bash
docker exec -it pph2 bash

apt-get update
apt-get install -y build-essential curl wget libexpat1-dev
# Install perlbrew (similar to conda) # https://metacpan.org/pod/App::perlbrew
curl -L https://install.perlbrew.pl | bash
source ~/perl5/perlbrew/etc/bashrc
cd ~/perl5/perlbrew/build
# Install cpanm (similar to pip)
perlbrew install-cpanm
# Install patchperl
cpanm Devel::PatchPerl
# Install perl-5.14.3 using patchperl
wget https://www.cpan.org/src/5.0/perl-5.14.3.tar.gz
tar -xzf perl-5.14.3.tar.gz
cd perl-5.14.3
patchperl
# sh Configure -de -Dusethreads
sh Configure -de -Dusethreads -Dprefix=/root/localperl514
make
make test
make install
# Instal perl modules (1) XML::Simple (2) LWP::Simple (3) DBD::SQLite
/usr/local/bin/perl /root/perl5/perlbrew/bin/cpanm XML::Simple LWP::Simple DBD::SQLite
/usr/local/bin/perl /root/perl5/perlbrew/bin/cpanm --notest Bio::Perl

export PATH=/usr/bin:$PATH
hash -r

# search from here to find BioPerl version https://www.cpan.org/authors/id/C/CJ/CJFIELDS/
/usr/local/bin/perl /root/perl5/perlbrew/bin/cpanm Bio::Perl@1.007004 


# Copy databases from A100
scp yang_loci@140.114.97.192:/mnt/nas_1/YangLab/project/polyphen-2.2.3-databases-pdb-dssp-2021_05.tar.bz2 .
scp yang_loci@140.114.97.192:/mnt/nas_1/YangLab/project/polyphen-2.2.3-databases-2021_05.tar.bz2 .
scp yang_loci@140.114.97.192:/mnt/nas_1/YangLab/project/polyphen-2.2.3-alignments-mlc-2011_12.tar.bz2 .
scp yang_loci@140.114.97.192:/mnt/nas_1/YangLab/project/polyphen-2.2.3-alignments-multiz-2009_10.tar.bz2 .
# Or download from website
# polyphen-2.2.3-alignments-multiz-2009_10.tar.bz2 OR Alignments (Multiz 2009/10) – version 2.2.2 (1.7 GB)
aria2c -x 8 -s 8 -k 1M 'https://dataverse.harvard.edu/api/access/datafile/10794995' # version 2.2.3
# polyphen-2.2.3-databases-2021_05.tar.bz2 OR Databases file (6.5 GB)
wget 'https://dataverse.harvard.edu/api/access/datafile/10211975'
aria2c -x 8 -s 8 -k 1M https://genetics.bwh.harvard.edu/downloads/pph2/bundled/polyphen-2.2.3-databases-2021_05.tar.bz2
# polyphen-2.2.3-databases-pdb-dssp-2021_05.tar.bz2 OR Databases with PDB/DSSP (38 GB)
wget 'https://dataverse.harvard.edu/api/access/datafile/10212039'
aria2c -x 8 -s 8 -k 1M https://genetics.bwh.harvard.edu/downloads/pph2/bundled/polyphen-2.2.3-databases-pdb-dssp-2021_05.tar.bz2
# polyphen-2.2.3-alignments-mlc-2011_12.tar.bz2
aria2c -x 8 -s 8 -k 1M https://dataverse.harvard.edu/api/access/datafile/10794996 # version 2.2.3

tar vxjf polyphen-2.2.3-databases-2021_05.tar.bz2
tar vxjf polyphen-2.2.3-databases-pdb-dssp-2021_05.tar.bz2
tar vxjf polyphen-2.2.3-alignments-mlc-2011_12.tar.bz2
tar vxjf polyphen-2.2.3-alignments-multiz-2009_10.tar.bz2

# Download the NCBI BLAST+ tools, version  2.2.26
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.2.26/ncbi-blast-2.2.26+-x64-linux.tar.gz
tar vxzf ncbi-blast-2.2.26+-x64-linux.tar.gz
mv ncbi-blast-2.2.26+/* polyphen-2.2.3/blast/

#################################################################################
# Maybe no need
# Download and install the UniRef100 nonredundant protein sequence database
cd $PPH/nrdb
wget ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref100/uniref100.fasta.gz
#################################################################################
# Install pph2, only need to do once.
export PPH=/pph2/polyphen-2.2.3
export PATH="$PATH:$PPH/bin"

echo $PPH
which perl
cd $PPH/src
make download
cd $PPH/src
make clean
make
make install
cd $PPH
yes | ./configure





export PATH=/usr/bin/perl/
/opt/perl514/bin:$PATH


apt-get install -y libdb-dev
apt-get install -y libxml2-dev libxslt1-dev zlib1g-dev libssl-dev
apt-get install libxml-simple-perl libwww-perl libdbd-sqlite3-perl
apt-get install libcgi-pm-perl
apt-get install bioperl


perl -v & which perl
perl -e '
for my $m (qw(XML::Simple LWP::Simple DBD::SQLite)) {
    eval "use $m; 1"
      ? print "$m: installed\n"
      : print "$m: NOT installed\n";
}
'
bin/run_pph.pl sets/test.input



/usr/local/bin/perl /root/perl5/perlbrew/bin/cpanm XML::LibXML IO::Socket::SSL LWP::Protocol::https DB_File
/usr/local/bin/perl /root/perl5/perlbrew/bin/cpanm Bio::Seq
/usr/local/bin/perl /root/perl5/perlbrew/bin/cpanm Bio::Search::HSP::GenericHSP::SUPER

/usr/local/bin/perl /root/perl5/perlbrew/bin/cpanm Bio::DB::BioFetch
/usr/local/bin/perl /root/perl5/perlbrew/bin/cpanm Bio::Perl --force
DBD::SQLite::db




apt-get install -y libexpat1-dev
/usr/local/bin/
perl -e '
for my $m (qw(XML::Simple LWP::Simple DBD::SQLite)) {
    eval "use $m; 1"
      ? print "$m: installed\n"
      : print "$m: NOT installed\n";
}
'
bin/run_pph.pl sets/test.input


# Construct Docker container
# We will use ubuntu:16.04
docker pull ubuntu:16.04

# ✅ Step 1 — start container
docker rm -f pph2
docker run -w /pph2 -dit --name pph2 \
  -v /home/loci/main/tandem_website_dev/tandem/PolyPhen-2:/pph2 \
  ubuntu:16.04 bash
docker exec -it pph2 bash



# Step 3 — install EVERYTHING in one go
apt-get update
apt-get install -y perl
\
    build-essential \
    wget \
    curl \
    perl \
    libsqlite3-dev \
    libssl-dev \
    zlib1g-dev \
    libexpat1-dev \
    ca-certificates
# To find a Perl module that's provided as an Ubuntu package:
apt-cache search perl <module-name>
apt-cache search perl XML

# We install perl 5.14.3 (install from source)
cd /root
wget https://www.cpan.org/src/5.0/perl-5.14.3.tar.gz
tar -xzf perl-5.14.3.tar.gz
cd perl-5.14.3

./Configure -des -Dprefix=/opt/perl514
make -j$(nproc)
make install
# use ONLY this Perl
export PATH=/opt/perl514/bin:$PATH
hash -r

which perl
perl -v
# install cpanminus (IMPORTANT)
curl -L https://cpanmin.us | /opt/perl514/bin/perl - App::cpanminus
/opt/perl514/bin/cpanm XML::Simple


curl -L https://cpanmin.us | perl -
cpanm --notest XML::Simple
cpanm --notest LWP::Simple
cpanm --notest DBD::SQLite
cpanm --notest Bio::Perl
perl -MXML::Simple -e 'print "OK\n";'
perl -MLWP::Simple -e 'print "OK\n";'
perl -MDBD::SQLite -e 'print "OK\n";'
perl -MBio::Tree::Statistics -e 'print "OK\n";'


# We install perl 5.14.3 (install from source)
wget https://www.cpan.org/src/5.0/perl-5.14.3.tar.gz
tar -xzf perl-5.14.3.tar.gz
cd perl-5.14.3
./Configure -des -Dprefix=$HOME/localperl
make
make test
make install

echo 'export PATH=$HOME/localperl/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Set up environment variables
cat >> ~/.bashrc <<'EOF'
export PPH=/pph2/polyphen-2.2.3
export PATH="$PATH:$PPH/bin"
EOF
source ∼/.bashrc
source ~/.bashrc

# Install pph2
echo $PPH
which perl
cd $PPH/src
make download
cd $PPH/src
make clean
make
make install
cd $PPH
yes | ./configure

# Test system
bin/run_pph.pl sets/test.input 1>test.pph.output 2>test.pph.log
bin/run_weka.pl test.pph.output >test.humdiv.output
bin/run_weka.pl -l models/HumVar.UniRef100.NBd.f11.model test.pph.output >test.humvar.output
diff test.humdiv.output sets/test.humdiv.output
diff test.humvar.output sets/test.humvar.output


bin/run_pph.pl \
  -c nonhuman_config \
  -s /pph2/Q46897.fasta \
  -d /pph2/Q46897_scratch \
  /pph2/SAVs.txt \
  > Q46897_pph2.tsv \
  2> Q46897_pph2.log
```