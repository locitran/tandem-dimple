/*=========================================================
**
** This program is designed for running RONNv3.2 for detecting
** disordered residues in proteins.
** Date:	March 2012
** Designer:	Zheng Rong Yang (Exeter University)
** Programmer:	Varun Ramraj (University of Oxford)
** Contact:	varun@strubi.ox.ac.uk
**
** Substantial modifications to speed up code as well as facilitate wrapping
** with Python (or any other language). Can now also handle multiple
** FASTA-formatted records in an input file. Path to `data` directory must be
** specified as a command-line argument if not located at `../data` with respect
** to the executable file.
** Date:    September 2016
** Author:  Shyam Saladi (California Institute of Technology)
** Contact: saladi@caltech.edu
**
==========================================================*/

// make accessible from Python
#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

// Number of models to use
#define MODCNT 10

#include <string>
#include <limits.h>
#include <unistd.h>

#include "include/callBBF.h"

std::vector<RONNModel> models;


std::string getexepath()
// find current directory of executable on Linux
// http://stackoverflow.com/a/19535628/2320823
{
  char result[ PATH_MAX ];
  ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
  return std::string( result, (count > 0) ? count : 0 );
}


int read_all_models(std::string prefix, float disorder_weight)
/*****
	disorder_weight: VERY IMPORTANT. CONTROLS COST FUNCTION
*****/
{
    FILE *fp;
    std::string fModel, fPDF;
    std::stringstream *convert;

	for(int m = 0; m < MODCNT; m++)
	{
		convert = new std::stringstream();
		(*convert) << m;
		std::string mString = convert->str();
		fModel = prefix + "/c" + mString + "/model.rec";

		if(!(fp = fopen(fModel.c_str(), "r")))
		{
			std::cerr << "Can't find " << fModel << std::endl;
			return -1;
		}
		fclose(fp);

		fPDF = prefix + "/c" + mString + "/pdfs.rec";
		if(!(fp = fopen(fPDF.c_str(), "r")))
		{
			std::cerr << "Can't find " << fPDF << std::endl;
			return -1;
		}
		fclose(fp);

        models.push_back(read_model_data(fModel, fPDF, disorder_weight));
    }
    delete convert;

    return 0;
}


DLLEXPORT
int read_all_models(const char *prefix, float disorder_weight) {
    std::string p(prefix);
    return read_all_models(p, disorder_weight);
}

DLLEXPORT
int predict_seq(const char *query, double *scores, bool print_output) {
    // scores is not cleared before calculation
    // if not all 0.0, will be added to
    for(int m = 0; m < MODCNT; m++) {
        predict_model(query, &models[m], scores);
    }

    for(unsigned int i = 0; i < strlen(query); i++) {
        scores[i] = scores[i]/(double)MODCNT;
        if(print_output)
             printf("%c\t%f\n", query[i], scores[i]);
    }

    return 0;
}


int predict_seq(std::string query, bool print_output) {
    std::vector<double> s(query.length(), 0.0);
    double* scores = &s[0];
    predict_seq(query.c_str(), scores, print_output);
    return 0;
}


int main(int argc, char *argv[])
{
	std::string query, line, prefix;

    if(argc == 2)
        prefix = argv[1];
    else if(argc > 2)
    {
        std::cerr << "Command line argument not recognized";
        return 1;
    }
    else
        prefix = getexepath() + "/../data/";

    // read the models
    int status = read_all_models(prefix, 0.53);
    if(status != 0)
        return(status);

    // read and predict sequences
    while(std::getline(std::cin, line).good()){
        if(line[0] == '>')
        {
            if(query.length() > 1)             // OK if not the first sequence
                predict_seq(query, 1);      // predict and print scores
            std::cout << line << std::endl; // print header
            query = "";
        }
        else
            query = query + line;
    }
    predict_seq(query, 1); // final sequence

	return 0;
}
