FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# # System packages
# RUN apt-get update && apt-get install -y build-essential curl wget libexpat1-dev
# # Install perlbrew (similar to conda)
# RUN curl -L https://install.perlbrew.pl | bash
# RUN source ~/perl5/perlbrew/etc/bashrc
# RUN cd ~/perl5/perlbrew/build
# # Install cpanm (similar to pip)
# RUN perlbrew install-cpanm
# # Install patchperl
# RUN cpanm Devel::PatchPerl
# # Install perl-5.14.3 using patchperl
# RUN wget https://www.cpan.org/src/5.0/perl-5.14.3.tar.gz
# RUN tar -xzf perl-5.14.3.tar.gz
# RUN cd perl-5.14.3
# RUN patchperl
# # sh Configure -de -Dusethreads
# RUN sh Configure -de -Dusethreads -Dprefix=/root/localperl514
# RUN make
# RUN make test
# RUN make install
# # Instal perl modules (1) XML::Simple (2) LWP::Simple (3) DBD::SQLite
# RUN perl /root/perl5/perlbrew/bin/cpanm XML::Simple LWP::Simple DBD::SQLite
# # Instal perl modules (1) XML::Simple (2) LWP::Simple (3) DBD::SQLite
# RUN /usr/local/bin/perl /root/perl5/perlbrew/bin/cpanm --notest XML::Simple LWP::Simple DBD::SQLite

###################### Install using root perl ######################
RUN apt-get update -qq && apt-get install -y -qq tzdata \
        build-essential curl wget python3-pip \
        libdb-dev libxml2-dev libxslt1-dev zlib1g-dev libssl-dev \
        libxml-simple-perl libwww-perl libdbd-sqlite3-perl \
        libcgi-pm-perl bioperl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# bin/run_pph.pl sets/test.input 1>test.pph.output 2>test.pph.log
RUN pip3 install --no-cache-dir flask

ENV PPH=/pph2/polyphen-2.2.3
WORKDIR /pph2/polyphen-2.2.3

EXPOSE 5001

CMD ["python3", "/pph2/main.py"]
